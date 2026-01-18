import enum
import os
from typing import Optional
from huggingface_hub.utils import insecure_hashlib
from .. import config
from .logging import get_logger
class UnexpectedDownloadedFile(ChecksumVerificationException):
    """Some downloaded files were not expected."""