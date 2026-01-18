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
def _get_make_request_args_put_object(self, request_type, call_args, coordinator, future, on_done_before_calls, on_done_after_calls):
    send_filepath = None
    if isinstance(call_args.fileobj, str):
        send_filepath = call_args.fileobj
        data_len = self._os_utils.get_file_size(send_filepath)
        call_args.extra_args['ContentLength'] = data_len
    else:
        call_args.extra_args['Body'] = call_args.fileobj
    checksum_algorithm = call_args.extra_args.pop('ChecksumAlgorithm', 'CRC32').upper()
    checksum_config = awscrt.s3.S3ChecksumConfig(algorithm=awscrt.s3.S3ChecksumAlgorithm[checksum_algorithm], location=awscrt.s3.S3ChecksumLocation.TRAILER)
    call_args.extra_args['ContentMD5'] = 'override-to-be-removed'
    make_request_args = self._default_get_make_request_args(request_type=request_type, call_args=call_args, coordinator=coordinator, future=future, on_done_before_calls=on_done_before_calls, on_done_after_calls=on_done_after_calls)
    make_request_args['send_filepath'] = send_filepath
    make_request_args['checksum_config'] = checksum_config
    return make_request_args