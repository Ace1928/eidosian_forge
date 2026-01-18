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
class MockFileSystem(pa_fs.FileSystem):

    @staticmethod
    def from_uri(path):
        print('Using pyarrow filesystem')
        to_local = pathlib.Path(path.replace('gs://', '')).absolute().as_uri()
        return pa_fs.LocalFileSystem(to_local)