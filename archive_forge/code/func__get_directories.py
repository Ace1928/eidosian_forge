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
def _get_directories(list_of_sources):
    direcs = []
    for f in list_of_sources:
        d = os.path.split(f)
        if d[0] != '' and (not d[0] in direcs):
            direcs.append(d[0])
    return direcs