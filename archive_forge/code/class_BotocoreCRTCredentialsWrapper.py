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
class BotocoreCRTCredentialsWrapper:

    def __init__(self, resolved_botocore_credentials):
        self._resolved_credentials = resolved_botocore_credentials

    def __call__(self):
        credentials = self._get_credentials().get_frozen_credentials()
        return AwsCredentials(credentials.access_key, credentials.secret_key, credentials.token)

    def to_crt_credentials_provider(self):
        return AwsCredentialsProvider.new_delegate(self)

    def _get_credentials(self):
        if self._resolved_credentials is None:
            raise NoCredentialsError()
        return self._resolved_credentials