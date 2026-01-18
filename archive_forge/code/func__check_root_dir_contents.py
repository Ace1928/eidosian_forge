from datetime import datetime, timezone, timedelta
import gzip
import os
import pathlib
import subprocess
import sys
import pytest
import weakref
import pyarrow as pa
from pyarrow.tests.test_io import assert_file_not_found
from pyarrow.tests.util import (_filesystem_uri, ProxyHandler,
from pyarrow.fs import (FileType, FileInfo, FileSelector, FileSystem,
def _check_root_dir_contents(config):
    fs = config['fs']
    pathfn = config['pathfn']
    d = pathfn('directory/')
    nd = pathfn('directory/nested/')
    fs.create_dir(nd)
    with pytest.raises(pa.ArrowInvalid):
        fs.delete_dir_contents('')
    with pytest.raises(pa.ArrowInvalid):
        fs.delete_dir_contents('/')
    with pytest.raises(pa.ArrowInvalid):
        fs.delete_dir_contents('//')
    fs.delete_dir_contents('', accept_root_dir=True)
    fs.delete_dir_contents('/', accept_root_dir=True)
    fs.delete_dir_contents('//', accept_root_dir=True)
    with pytest.raises(pa.ArrowIOError):
        fs.delete_dir(d)