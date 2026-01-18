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
def _get_make_request_args_get_object(self, request_type, call_args, coordinator, future, on_done_before_calls, on_done_after_calls):
    recv_filepath = None
    on_body = None
    checksum_config = awscrt.s3.S3ChecksumConfig(validate_response=True)
    if isinstance(call_args.fileobj, str):
        final_filepath = call_args.fileobj
        recv_filepath = self._os_utils.get_temp_filename(final_filepath)
        on_done_before_calls.append(RenameTempFileHandler(coordinator, final_filepath, recv_filepath, self._os_utils))
    else:
        on_body = OnBodyFileObjWriter(call_args.fileobj)
    make_request_args = self._default_get_make_request_args(request_type=request_type, call_args=call_args, coordinator=coordinator, future=future, on_done_before_calls=on_done_before_calls, on_done_after_calls=on_done_after_calls)
    make_request_args['recv_filepath'] = recv_filepath
    make_request_args['on_body'] = on_body
    make_request_args['checksum_config'] = checksum_config
    return make_request_args