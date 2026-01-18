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
def _convert_to_crt_http_request(self, botocore_http_request):
    crt_request = self._crt_request_from_aws_request(botocore_http_request)
    if crt_request.headers.get('host') is None:
        url_parts = urlsplit(botocore_http_request.url)
        crt_request.headers.set('host', url_parts.netloc)
    if crt_request.headers.get('Content-MD5') is not None:
        crt_request.headers.remove('Content-MD5')
    if crt_request.headers.get('Content-Length') is None:
        if botocore_http_request.body is None:
            crt_request.headers.add('Content-Length', '0')
    if crt_request.headers.get('Transfer-Encoding') is not None:
        crt_request.headers.remove('Transfer-Encoding')
    return crt_request