import logging
import threading
import heapq
from botocore.compat import six
from s3transfer.compat import seekable
from s3transfer.exceptions import RetriesExceededError
from s3transfer.futures import IN_MEMORY_DOWNLOAD_TAG
from s3transfer.utils import S3_RETRYABLE_DOWNLOAD_ERRORS
from s3transfer.utils import get_callbacks
from s3transfer.utils import invoke_progress_callbacks
from s3transfer.utils import calculate_num_parts
from s3transfer.utils import calculate_range_parameter
from s3transfer.utils import FunctionContainer
from s3transfer.utils import CountCallbackInvoker
from s3transfer.utils import StreamReaderProgress
from s3transfer.utils import DeferredOpenFile
from s3transfer.tasks import Task
from s3transfer.tasks import SubmissionTask
class GetObjectTask(Task):

    def _main(self, client, bucket, key, fileobj, extra_args, callbacks, max_attempts, download_output_manager, io_chunksize, start_index=0, bandwidth_limiter=None):
        """Downloads an object and places content into io queue

        :param client: The client to use when calling GetObject
        :param bucket: The bucket to download from
        :param key: The key to download from
        :param fileobj: The file handle to write content to
        :param exta_args: Any extra arguements to include in GetObject request
        :param callbacks: List of progress callbacks to invoke on download
        :param max_attempts: The number of retries to do when downloading
        :param download_output_manager: The download output manager associated
            with the current download.
        :param io_chunksize: The size of each io chunk to read from the
            download stream and queue in the io queue.
        :param start_index: The location in the file to start writing the
            content of the key to.
        :param bandwidth_limiter: The bandwidth limiter to use when throttling
            the downloading of data in streams.
        """
        last_exception = None
        for i in range(max_attempts):
            try:
                current_index = start_index
                response = client.get_object(Bucket=bucket, Key=key, **extra_args)
                streaming_body = StreamReaderProgress(response['Body'], callbacks)
                if bandwidth_limiter:
                    streaming_body = bandwidth_limiter.get_bandwith_limited_stream(streaming_body, self._transfer_coordinator)
                chunks = DownloadChunkIterator(streaming_body, io_chunksize)
                for chunk in chunks:
                    if not self._transfer_coordinator.done():
                        self._handle_io(download_output_manager, fileobj, chunk, current_index)
                        current_index += len(chunk)
                    else:
                        return
                return
            except S3_RETRYABLE_DOWNLOAD_ERRORS as e:
                logger.debug('Retrying exception caught (%s), retrying request, (attempt %s / %s)', e, i, max_attempts, exc_info=True)
                last_exception = e
                invoke_progress_callbacks(callbacks, start_index - current_index)
                continue
        raise RetriesExceededError(last_exception)

    def _handle_io(self, download_output_manager, fileobj, chunk, index):
        download_output_manager.queue_file_io_task(fileobj, chunk, index)