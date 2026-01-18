import hashlib
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import pytest
from ..core import create, Pooch, retrieve, download_action, stream_download
from ..utils import get_logger, temporary_file, os_cache
from ..hashes import file_hash, hash_matches
from .. import core
from ..downloaders import HTTPDownloader, FTPDownloader
from .utils import (
class FakeSleep:
    """Create a fake version of sleep that logs the specified times"""

    def __init__(self):
        self.times = []

    def sleep(self, secs):
        """Store the time and doesn't sleep"""
        self.times.append(secs)