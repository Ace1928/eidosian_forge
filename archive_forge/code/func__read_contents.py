import datetime
import warnings
import random
import string
import tempfile
import os
import contextlib
import json
import urllib.request
import hashlib
import time
import subprocess as sp
import multiprocessing as mp
import platform
import pickle
import zipfile
import re
import av
import pytest
from tensorflow.io import gfile
import imageio
import numpy as np
import blobfile as bf
from blobfile import _ops as ops, _azure as azure, _common as common
def _read_contents(path):
    if '.blob.core.windows.net' in path:
        with tempfile.TemporaryDirectory() as tmpdir:
            assert isinstance(tmpdir, str)
            account, container, blob = azure.split_path(path)
            filepath = os.path.join(tmpdir, 'tmp')
            sp.run(['az', 'storage', 'blob', 'download', '--account-name', account, '--container-name', container, '--name', blob, '--file', filepath], check=True, shell=platform.system() == 'Windows', stdout=sp.DEVNULL, stderr=sp.DEVNULL)
            with open(filepath, 'rb') as f:
                return f.read()
    else:
        with gfile.GFile(path, 'rb') as f:
            return f.read()