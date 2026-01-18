import json
import os.path as osp
from itertools import filterfalse
from .jlpmapp import HERE
def _is_lab_package(name):
    """Whether a package name is in the lab namespace"""
    return name.startswith('@jupyterlab/')