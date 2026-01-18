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
def _should_use_crt(config):
    if HAS_CRT and has_minimum_crt_version((0, 19, 18)):
        is_optimized_instance = awscrt.s3.is_optimized_for_system()
    else:
        is_optimized_instance = False
    pref_transfer_client = config.preferred_transfer_client.lower()
    if is_optimized_instance and pref_transfer_client == constants.AUTO_RESOLVE_TRANSFER_CLIENT:
        logger.debug('Attempting to use CRTTransferManager. Config settings may be ignored.')
        return True
    logger.debug(f'Opting out of CRT Transfer Manager. Preferred client: {pref_transfer_client}, CRT available: {HAS_CRT}, Instance Optimized: {is_optimized_instance}.')
    return False