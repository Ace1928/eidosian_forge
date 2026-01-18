import glob
import fnmatch
import string
import json
import os
import os.path as op
import shutil
import subprocess
import re
import copy
import tempfile
from os.path import join, dirname
from warnings import warn
from .. import config, logging
from ..utils.filemanip import (
from ..utils.misc import human_order_sorted, str2bool
from .base import (
def _get_dst(self, src):
    src = src.rstrip(os.path.sep)
    path, fname = os.path.split(src)
    if self.inputs.parameterization:
        dst = path
        if isdefined(self.inputs.strip_dir):
            dst = dst.replace(self.inputs.strip_dir, '')
        folders = [folder for folder in dst.split(os.path.sep) if folder.startswith('_')]
        dst = os.path.sep.join(folders)
        if fname:
            dst = os.path.join(dst, fname)
    elif fname:
        dst = fname
    else:
        dst = path.split(os.path.sep)[-1]
    if dst[0] == os.path.sep:
        dst = dst[1:]
    return dst