import functools
import itertools
import os
import shutil
import subprocess
import sys
import textwrap
import threading
import time
import warnings
import zipfile
from hashlib import md5
from xml.etree import ElementTree
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
import nltk
def _pkg_status(self, info, filepath):
    if not os.path.exists(filepath):
        return self.NOT_INSTALLED
    try:
        filestat = os.stat(filepath)
    except OSError:
        return self.NOT_INSTALLED
    if filestat.st_size != int(info.size):
        return self.STALE
    if md5_hexdigest(filepath) != info.checksum:
        return self.STALE
    if filepath.endswith('.zip'):
        unzipdir = filepath[:-4]
        if not os.path.exists(unzipdir):
            return self.INSTALLED
        if not os.path.isdir(unzipdir):
            return self.STALE
        unzipped_size = sum((os.stat(os.path.join(d, f)).st_size for d, _, files in os.walk(unzipdir) for f in files))
        if unzipped_size != info.unzipped_size:
            return self.STALE
    return self.INSTALLED