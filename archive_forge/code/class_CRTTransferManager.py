import logging
import threading
from io import BytesIO
import awscrt.http
import awscrt.s3
import botocore.awsrequest
import botocore.session
from awscrt.auth import (
from awscrt.io import (
from awscrt.s3 import S3Client, S3RequestTlsMode, S3RequestType
from botocore import UNSIGNED
from botocore.compat import urlsplit
from botocore.config import Config
from botocore.exceptions import NoCredentialsError
from s3transfer.constants import MB
from s3transfer.exceptions import TransferNotDoneError
from s3transfer.futures import BaseTransferFuture, BaseTransferMeta
from s3transfer.utils import (
class CRTTransferManager:

    def __init__(self, crt_s3_client, crt_request_serializer, osutil=None):
        """A transfer manager interface for Amazon S3 on CRT s3 client.

        :type crt_s3_client: awscrt.s3.S3Client
        :param crt_s3_client: The CRT s3 client, handling all the
            HTTP requests and functions under then hood

        :type crt_request_serializer: s3transfer.crt.BaseCRTRequestSerializer
        :param crt_request_serializer: Serializer, generates unsigned crt HTTP
            request.

        :type osutil: s3transfer.utils.OSUtils
        :param osutil: OSUtils object to use for os-related behavior when
            using with transfer manager.
        """
        if osutil is None:
            self._osutil = OSUtils()
        self._crt_s3_client = crt_s3_client
        self._s3_args_creator = S3ClientArgsCreator(crt_request_serializer, self._osutil)
        self._crt_exception_translator = crt_request_serializer.translate_crt_exception
        self._future_coordinators = []
        self._semaphore = threading.Semaphore(128)
        self._id_counter = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, *args):
        cancel = False
        if exc_type:
            cancel = True
        self._shutdown(cancel)

    def download(self, bucket, key, fileobj, extra_args=None, subscribers=None):
        if extra_args is None:
            extra_args = {}
        if subscribers is None:
            subscribers = {}
        callargs = CallArgs(bucket=bucket, key=key, fileobj=fileobj, extra_args=extra_args, subscribers=subscribers)
        return self._submit_transfer('get_object', callargs)

    def upload(self, fileobj, bucket, key, extra_args=None, subscribers=None):
        if extra_args is None:
            extra_args = {}
        if subscribers is None:
            subscribers = {}
        self._validate_checksum_algorithm_supported(extra_args)
        callargs = CallArgs(bucket=bucket, key=key, fileobj=fileobj, extra_args=extra_args, subscribers=subscribers)
        return self._submit_transfer('put_object', callargs)

    def delete(self, bucket, key, extra_args=None, subscribers=None):
        if extra_args is None:
            extra_args = {}
        if subscribers is None:
            subscribers = {}
        callargs = CallArgs(bucket=bucket, key=key, extra_args=extra_args, subscribers=subscribers)
        return self._submit_transfer('delete_object', callargs)

    def shutdown(self, cancel=False):
        self._shutdown(cancel)

    def _validate_checksum_algorithm_supported(self, extra_args):
        checksum_algorithm = extra_args.get('ChecksumAlgorithm')
        if checksum_algorithm is None:
            return
        supported_algorithms = list(awscrt.s3.S3ChecksumAlgorithm.__members__)
        if checksum_algorithm.upper() not in supported_algorithms:
            raise ValueError(f'ChecksumAlgorithm: {checksum_algorithm} not supported. Supported algorithms are: {supported_algorithms}')

    def _cancel_transfers(self):
        for coordinator in self._future_coordinators:
            if not coordinator.done():
                coordinator.cancel()

    def _finish_transfers(self):
        for coordinator in self._future_coordinators:
            coordinator.result()

    def _wait_transfers_done(self):
        for coordinator in self._future_coordinators:
            coordinator.wait_until_on_done_callbacks_complete()

    def _shutdown(self, cancel=False):
        if cancel:
            self._cancel_transfers()
        try:
            self._finish_transfers()
        except KeyboardInterrupt:
            self._cancel_transfers()
        except Exception:
            pass
        finally:
            self._wait_transfers_done()

    def _release_semaphore(self, **kwargs):
        self._semaphore.release()

    def _submit_transfer(self, request_type, call_args):
        on_done_after_calls = [self._release_semaphore]
        coordinator = CRTTransferCoordinator(transfer_id=self._id_counter, exception_translator=self._crt_exception_translator)
        components = {'meta': CRTTransferMeta(self._id_counter, call_args), 'coordinator': coordinator}
        future = CRTTransferFuture(**components)
        afterdone = AfterDoneHandler(coordinator)
        on_done_after_calls.append(afterdone)
        try:
            self._semaphore.acquire()
            on_queued = self._s3_args_creator.get_crt_callback(future, 'queued')
            on_queued()
            crt_callargs = self._s3_args_creator.get_make_request_args(request_type, call_args, coordinator, future, on_done_after_calls)
            crt_s3_request = self._crt_s3_client.make_request(**crt_callargs)
        except Exception as e:
            coordinator.set_exception(e, True)
            on_done = self._s3_args_creator.get_crt_callback(future, 'done', after_subscribers=on_done_after_calls)
            on_done(error=e)
        else:
            coordinator.set_s3_request(crt_s3_request)
        self._future_coordinators.append(coordinator)
        self._id_counter += 1
        return future