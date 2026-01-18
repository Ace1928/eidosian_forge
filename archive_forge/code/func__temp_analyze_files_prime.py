import os
import time
from pathlib import Path
from unittest import mock, SkipTest
import pytest
from ...testing import TempFATFS
from ...utils.filemanip import (
@pytest.fixture()
def _temp_analyze_files_prime(tmpdir):
    """Generate temporary analyze file pair."""
    orig_img = tmpdir.join('orig_prime.img')
    orig_hdr = tmpdir.join('orig_prime.hdr')
    orig_img.open('w+').close()
    orig_hdr.open('w+').close()
    return (orig_img.strpath, orig_hdr.strpath)