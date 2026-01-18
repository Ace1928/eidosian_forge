from collections import OrderedDict
import io
import warnings
from shutil import copytree
import numpy as np
import pytest
import pyarrow as pa
from pyarrow import fs
from pyarrow.filesystem import LocalFileSystem, FileSystem
from pyarrow.tests import util
from pyarrow.tests.parquet.common import (_check_roundtrip, _roundtrip_table,
class BogusFile(io.BytesIO):

    def read(self, *args):
        raise ZeroDivisionError('zorglub')

    def seek(self, *args):
        raise ZeroDivisionError('zorglub')