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
def _dist_info_files(whl_zip):
    """Identify the .dist-info folder inside a wheel ZipFile."""
    res = []
    for path in whl_zip.namelist():
        m = re.match('[^/\\\\]+-[^/\\\\]+\\.dist-info/', path)
        if m:
            res.append(path)
    if res:
        return res
    raise Exception('No .dist-info folder found in wheel')