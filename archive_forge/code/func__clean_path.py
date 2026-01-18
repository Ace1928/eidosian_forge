import os
import sys
import json
import string
import shutil
import logging
import coloredlogs
import fire
import requests
from .._utils import run_command_with_process, compute_md5, job
@staticmethod
def _clean_path(path):
    if os.path.exists(path):
        logger.warning('🚨 %s already exists, remove it!', path)
        try:
            if os.path.isfile(path):
                os.remove(path)
            if os.path.isdir(path):
                shutil.rmtree(path)
        except OSError:
            sys.exit(1)
    else:
        logger.warning("🚨 %s doesn't exist, no action taken", path)