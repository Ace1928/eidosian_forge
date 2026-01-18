import logging
import threading
from os import PathLike, fspath, getpid
from botocore.compat import HAS_CRT
from botocore.exceptions import ClientError
from s3transfer.exceptions import (
from s3transfer.futures import NonThreadedExecutor
from s3transfer.manager import TransferConfig as S3TransferConfig
from s3transfer.manager import TransferManager
from s3transfer.subscribers import BaseSubscriber
from s3transfer.utils import OSUtils
import boto3.s3.constants as constants
from boto3.exceptions import RetriesExceededError, S3UploadFailedError
def _create_default_transfer_manager(client, config, osutil):
    """Create the default TransferManager implementation for s3transfer."""
    executor_cls = None
    if not config.use_threads:
        executor_cls = NonThreadedExecutor
    return TransferManager(client, config, osutil, executor_cls)