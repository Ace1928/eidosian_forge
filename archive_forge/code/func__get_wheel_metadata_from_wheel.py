import json
import os
import os.path
import re
import shutil
import sys
import traceback
from glob import glob
from importlib import import_module
from os.path import join as pjoin
def _get_wheel_metadata_from_wheel(whl_basename, metadata_directory, config_settings):
    """Extract the metadata from a wheel.

    Fallback for when the build backend does not
    define the 'get_wheel_metadata' hook.
    """
    from zipfile import ZipFile
    with open(os.path.join(metadata_directory, WHEEL_BUILT_MARKER), 'wb'):
        pass
    whl_file = os.path.join(metadata_directory, whl_basename)
    with ZipFile(whl_file) as zipf:
        dist_info = _dist_info_files(zipf)
        zipf.extractall(path=metadata_directory, members=dist_info)
    return dist_info[0].split('/')[0]