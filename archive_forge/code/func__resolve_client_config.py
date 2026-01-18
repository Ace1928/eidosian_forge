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
def _resolve_client_config(self, session, client_kwargs):
    user_provided_config = None
    if session.get_default_client_config():
        user_provided_config = session.get_default_client_config()
    if 'config' in client_kwargs:
        user_provided_config = client_kwargs['config']
    client_config = Config(signature_version=UNSIGNED)
    if user_provided_config:
        client_config = user_provided_config.merge(client_config)
    client_kwargs['config'] = client_config
    client_kwargs['service_name'] = 's3'