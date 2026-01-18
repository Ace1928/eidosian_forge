import enum
import os
from typing import Optional
from huggingface_hub.utils import insecure_hashlib
from .. import config
from .logging import get_logger
class ExpectedMoreDownloadedFiles(ChecksumVerificationException):
    """Some files were supposed to be downloaded but were not."""