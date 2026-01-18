import argparse
import fnmatch
import re
import shutil
import sys
import os
from . import constants
from .cuda_to_hip_mappings import CUDA_TO_HIP_MAPPINGS
from .cuda_to_hip_mappings import MATH_TRANSPILATIONS
from typing import Dict, List, Iterator, Optional
from collections.abc import Mapping, Iterable
from enum import Enum
class GeneratedFileCleaner:
    """Context Manager to clean up generated files"""

    def __init__(self, keep_intermediates=False):
        self.keep_intermediates = keep_intermediates
        self.files_to_clean = set()
        self.dirs_to_clean = []

    def __enter__(self):
        return self

    def open(self, fn, *args, **kwargs):
        if not os.path.exists(fn):
            self.files_to_clean.add(os.path.abspath(fn))
        return open(fn, *args, **kwargs)

    def makedirs(self, dn, exist_ok=False):
        parent, n = os.path.split(dn)
        if not n:
            parent, n = os.path.split(parent)
        if parent and n and (not os.path.exists(parent)):
            self.makedirs(parent, exist_ok=True)
        if not os.path.isdir(dn) or not exist_ok:
            os.mkdir(dn)
            self.dirs_to_clean.append(os.path.abspath(dn))

    def __exit__(self, type, value, traceback):
        if not self.keep_intermediates:
            for f in self.files_to_clean:
                os.unlink(f)
            for d in self.dirs_to_clean[::-1]:
                os.rmdir(d)