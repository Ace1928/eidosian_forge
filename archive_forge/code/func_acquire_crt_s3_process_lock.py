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
def acquire_crt_s3_process_lock(name):
    global CRT_S3_PROCESS_LOCK
    if CRT_S3_PROCESS_LOCK is None:
        crt_lock = awscrt.s3.CrossProcessLock(name)
        try:
            crt_lock.acquire()
        except RuntimeError:
            return None
        CRT_S3_PROCESS_LOCK = crt_lock
    return CRT_S3_PROCESS_LOCK