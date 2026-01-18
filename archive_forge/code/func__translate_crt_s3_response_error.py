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
def _translate_crt_s3_response_error(self, s3_response_error):
    status_code = s3_response_error.status_code
    if status_code < 301:
        return None
    headers = {k: v for k, v in s3_response_error.headers}
    operation_name = s3_response_error.operation_name
    if operation_name is not None:
        service_model = self._client.meta.service_model
        shape = service_model.operation_model(operation_name).output_shape
    else:
        shape = None
    response_dict = {'headers': botocore.awsrequest.HeadersDict(headers), 'status_code': status_code, 'body': s3_response_error.body}
    parsed_response = self._client._response_parser.parse(response_dict, shape=shape)
    error_code = parsed_response.get('Error', {}).get('Code')
    error_class = self._client.exceptions.from_code(error_code)
    return error_class(parsed_response, operation_name=operation_name)