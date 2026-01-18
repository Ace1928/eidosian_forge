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
def _get_crt_throughput_target_gbps(provided_throughput_target_bytes=None):
    if provided_throughput_target_bytes is None:
        target_gbps = awscrt.s3.get_recommended_throughput_target_gbps()
        logger.debug('Recommended CRT throughput target in gbps: %s', target_gbps)
        if target_gbps is None:
            target_gbps = 10.0
    else:
        target_gbps = provided_throughput_target_bytes * 8 / 1000000000
    logger.debug('Using CRT throughput target in gbps: %s', target_gbps)
    return target_gbps