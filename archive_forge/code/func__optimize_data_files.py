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
def _optimize_data_files(self):
    data_dict = {}
    for p, files in self.data_files:
        if p not in data_dict:
            data_dict[p] = set()
        for f in files:
            data_dict[p].add(f)
    self.data_files[:] = [(p, list(files)) for p, files in data_dict.items()]