from math import log
import os
from os import path as op
import sys
import shutil
import time
from . import appdata_dir, resource_dirs
from . import StdoutProgressIndicator, urlopen
def _chunk_write(chunk, local_file, progress):
    """Write a chunk to file and update the progress bar"""
    local_file.write(chunk)
    progress.increase_progress(len(chunk))
    time.sleep(0)