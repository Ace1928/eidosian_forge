import os
from glob import glob
from shutil import rmtree
from string import Template
import numpy as np
from nibabel import load
from ... import LooseVersion
from ...utils.filemanip import simplify_list, ensure_list
from ...utils.misc import human_order_sorted
from ...external.due import BibTeX
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
def _get_func_files(self, session_info):
    """Returns functional files in the order of runs"""
    func_files = []
    for i, info in enumerate(session_info):
        func_files.insert(i, info['scans'])
    return func_files