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
class CRTTransferFuture(BaseTransferFuture):

    def __init__(self, meta=None, coordinator=None):
        """The future associated to a submitted transfer request via CRT S3 client

        :type meta: s3transfer.crt.CRTTransferMeta
        :param meta: The metadata associated to the transfer future.

        :type coordinator: s3transfer.crt.CRTTransferCoordinator
        :param coordinator: The coordinator associated to the transfer future.
        """
        self._meta = meta
        if meta is None:
            self._meta = CRTTransferMeta()
        self._coordinator = coordinator

    @property
    def meta(self):
        return self._meta

    def done(self):
        return self._coordinator.done()

    def result(self, timeout=None):
        self._coordinator.result(timeout)

    def cancel(self):
        self._coordinator.cancel()

    def set_exception(self, exception):
        """Sets the exception on the future."""
        if not self.done():
            raise TransferNotDoneError('set_exception can only be called once the transfer is complete.')
        self._coordinator.set_exception(exception, override=True)