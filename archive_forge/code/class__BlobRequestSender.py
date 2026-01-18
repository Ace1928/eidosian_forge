import contextlib
import functools
import time
import grpc
from google.protobuf import message
from tensorboard.compat.proto import graph_pb2
from tensorboard.compat.proto import summary_pb2
from tensorboard.compat.proto import types_pb2
from tensorboard.uploader.proto import write_service_pb2
from tensorboard.uploader import logdir_loader
from tensorboard.uploader import upload_tracker
from tensorboard.uploader import util
from tensorboard.backend import process_graph
from tensorboard.backend.event_processing import directory_loader
from tensorboard.backend.event_processing import event_file_loader
from tensorboard.backend.event_processing import io_wrapper
from tensorboard.plugins.graph import metadata as graphs_metadata
from tensorboard.util import grpc_util
from tensorboard.util import tb_logging
from tensorboard.util import tensor_util
class _BlobRequestSender:
    """Uploader for blob-type event data.

    Unlike the other types, this class does not accumulate events in batches;
    every blob is sent individually and immediately.  Nonetheless we retain
    the `add_event()`/`flush()` structure for symmetry.

    This class is not threadsafe. Use external synchronization if calling its
    methods concurrently.
    """

    def __init__(self, experiment_id, api, rpc_rate_limiter, max_blob_request_size, max_blob_size, tracker):
        if experiment_id is None:
            raise ValueError('experiment_id cannot be None')
        self._experiment_id = experiment_id
        self._api = api
        self._rpc_rate_limiter = rpc_rate_limiter
        self._max_blob_request_size = max_blob_request_size
        self._max_blob_size = max_blob_size
        self._tracker = tracker
        self._run_name = None
        self._event = None
        self._value = None
        self._metadata = None

    def _new_request(self):
        """Declares the previous event complete."""
        self._run_name = None
        self._event = None
        self._value = None
        self._metadata = None

    def add_event(self, run_name, event, value, metadata):
        """Attempts to add the given event to the current request.

        If the event cannot be added to the current request because the byte
        budget is exhausted, the request is flushed, and the event is added
        to the next request.
        """
        if self._value:
            raise RuntimeError('Tried to send blob while another is pending')
        self._run_name = run_name
        self._event = event
        self._value = value
        self._blobs = tensor_util.make_ndarray(self._value.tensor)
        if self._blobs.ndim == 1:
            self._metadata = metadata
            self.flush()
        else:
            logger.warning("A blob sequence must be represented as a rank-1 Tensor. Provided data has rank %d, for run %s, tag %s, step %s ('%s' plugin) .", self._blobs.ndim, run_name, self._value.tag, self._event.step, metadata.plugin_data.plugin_name)
            self._new_request()

    def flush(self):
        """Sends the current blob sequence fully, and clears it to make way for the next."""
        if self._value:
            blob_sequence_id = self._get_or_create_blob_sequence()
            logger.info('Sending %d blobs for sequence id: %s', len(self._blobs), blob_sequence_id)
            sent_blobs = 0
            for seq_index, blob in enumerate(self._blobs):
                self._rpc_rate_limiter.tick()
                with self._tracker.blob_tracker(len(blob)) as blob_tracker:
                    sent_blobs += self._send_blob(blob_sequence_id, seq_index, blob)
                    blob_tracker.mark_uploaded(bool(sent_blobs))
            logger.info('Sent %d of %d blobs for sequence id: %s', sent_blobs, len(self._blobs), blob_sequence_id)
        self._new_request()

    def _get_or_create_blob_sequence(self):
        request = write_service_pb2.GetOrCreateBlobSequenceRequest(experiment_id=self._experiment_id, run=self._run_name, tag=self._value.tag, step=self._event.step, final_sequence_length=len(self._blobs), metadata=self._metadata)
        util.set_timestamp(request.wall_time, self._event.wall_time)
        with _request_logger(request):
            try:
                response = grpc_util.call_with_retries(self._api.GetOrCreateBlobSequence, request)
                blob_sequence_id = response.blob_sequence_id
            except grpc.RpcError as e:
                if e.code() == grpc.StatusCode.NOT_FOUND:
                    raise ExperimentNotFoundError()
                logger.error('Upload call failed with error %s', e)
                raise
        return blob_sequence_id

    def _send_blob(self, blob_sequence_id, seq_index, blob):
        """Tries to send a single blob for a given index within a blob sequence.

        The blob will not be sent if it was sent already, or if it is too large.

        Returns:
          The number of blobs successfully sent (i.e., 1 or 0).
        """
        if len(blob) > self._max_blob_size:
            logger.warning('Blob too large; skipping.  Size %d exceeds limit of %d bytes.', len(blob), self._max_blob_size)
            return 0
        request_iterator = self._write_blob_request_iterator(blob_sequence_id, seq_index, blob)
        upload_start_time = time.time()
        count = 0
        try:
            for response in self._api.WriteBlob(request_iterator):
                count += 1
                pass
            upload_duration_secs = time.time() - upload_start_time
            logger.info('Upload for %d chunks totaling %d bytes took %.3f seconds (%.3f MB/sec)', count, len(blob), upload_duration_secs, len(blob) / upload_duration_secs / (1024 * 1024))
            return 1
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.ALREADY_EXISTS:
                logger.error('Attempted to re-upload existing blob.  Skipping.')
                return 0
            else:
                logger.info('WriteBlob RPC call got error %s', e)
                raise

    def _write_blob_request_iterator(self, blob_sequence_id, seq_index, blob):
        for offset in range(0, len(blob), self._max_blob_request_size):
            chunk = blob[offset:offset + self._max_blob_request_size]
            finalize_object = offset + self._max_blob_request_size >= len(blob)
            request = write_service_pb2.WriteBlobRequest(blob_sequence_id=blob_sequence_id, index=seq_index, data=chunk, offset=offset, crc32c=None, finalize_object=finalize_object, final_crc32c=None, blob_bytes=len(blob))
            yield request