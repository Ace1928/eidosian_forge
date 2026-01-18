import os
import shutil
import sys
from pathlib import Path
from typing import Union
from pyproj._datadir import (  # noqa: F401  pylint: disable=unused-import
from pyproj.exceptions import DataDirError

    The order of preference for the data directory is:

    1. The one set by pyproj.datadir.set_data_dir (if exists & valid)
    2. The internal proj directory (if exists & valid)
    3. The directory in PROJ_DATA (PROJ 9.1+) | PROJ_LIB (PROJ<9.1) (if exists & valid)
    4. The directory on sys.prefix (if exists & valid)
    5. The directory on the PATH (if exists & valid)

    Returns
    -------
    str:
        The valid data directory.

    