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
def _default_get_make_request_args(self, request_type, call_args, coordinator, future, on_done_before_calls, on_done_after_calls):
    make_request_args = {'request': self._request_serializer.serialize_http_request(request_type, future), 'type': getattr(S3RequestType, request_type.upper(), S3RequestType.DEFAULT), 'on_done': self.get_crt_callback(future, 'done', on_done_before_calls, on_done_after_calls), 'on_progress': self.get_crt_callback(future, 'progress')}
    if is_s3express_bucket(call_args.bucket):
        make_request_args['signing_config'] = AwsSigningConfig(algorithm=AwsSigningAlgorithm.V4_S3EXPRESS)
    return make_request_args