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
def _md5_hexdigest(fp):
    md5_digest = md5()
    while True:
        block = fp.read(1024 * 16)
        if not block:
            break
        md5_digest.update(block)
    return md5_digest.hexdigest()