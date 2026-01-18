from io import BytesIO
import os
import pathlib
import tarfile
import zipfile
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.util import _test_decorators as td
class MockGCSFileSystem(AbstractFileSystem):

    def open(self, path, mode='r', *args):
        if 'w' not in mode:
            raise FileNotFoundError
        return open(os.path.join(tmpdir, 'test.parquet'), mode, encoding='utf-8')