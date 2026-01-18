import os
import re
import sys
import copy
import glob
import atexit
import tempfile
import subprocess
import shutil
import multiprocessing
import textwrap
import importlib.util
from threading import local as tlocal
from functools import reduce
import distutils
from distutils.errors import DistutilsError
def _get_headers(directory_list):
    headers = []
    for d in directory_list:
        head = sorted_glob(os.path.join(d, '*.h'))
        headers.extend(head)
    return headers