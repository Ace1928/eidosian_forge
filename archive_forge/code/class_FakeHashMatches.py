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
class FakeHashMatches:
    """Create a fake version of hash_matches that fails n times"""

    def __init__(self, nfailures):
        self.nfailures = nfailures
        self.failed = 0

    def hash_matches(self, *args, **kwargs):
        """Fail n times before finally passing"""
        if self.failed < self.nfailures:
            self.failed += 1
            return hash_matches(args[0], 'bla', **kwargs)
        return hash_matches(*args, **kwargs)