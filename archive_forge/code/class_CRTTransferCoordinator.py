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
class CRTTransferCoordinator:
    """A helper class for managing CRTTransferFuture"""

    def __init__(self, transfer_id=None, s3_request=None, exception_translator=None):
        self.transfer_id = transfer_id
        self._exception_translator = exception_translator
        self._s3_request = s3_request
        self._lock = threading.Lock()
        self._exception = None
        self._crt_future = None
        self._done_event = threading.Event()

    @property
    def s3_request(self):
        return self._s3_request

    def set_done_callbacks_complete(self):
        self._done_event.set()

    def wait_until_on_done_callbacks_complete(self, timeout=None):
        self._done_event.wait(timeout)

    def set_exception(self, exception, override=False):
        with self._lock:
            if not self.done() or override:
                self._exception = exception

    def cancel(self):
        if self._s3_request:
            self._s3_request.cancel()

    def result(self, timeout=None):
        if self._exception:
            raise self._exception
        try:
            self._crt_future.result(timeout)
        except KeyboardInterrupt:
            self.cancel()
            self._crt_future.result(timeout)
            raise
        except Exception as e:
            self.handle_exception(e)
        finally:
            if self._s3_request:
                self._s3_request = None

    def handle_exception(self, exc):
        translated_exc = None
        if self._exception_translator:
            try:
                translated_exc = self._exception_translator(exc)
            except Exception as e:
                logger.debug('Unable to translate exception.', exc_info=e)
                pass
        if translated_exc is not None:
            raise translated_exc from exc
        else:
            raise exc

    def done(self):
        if self._crt_future is None:
            return False
        return self._crt_future.done()

    def set_s3_request(self, s3_request):
        self._s3_request = s3_request
        self._crt_future = self._s3_request.finished_future