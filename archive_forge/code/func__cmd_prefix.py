import os
from sys import platform
import shutil
from ... import logging, LooseVersion
from ...utils.filemanip import split_filename, fname_presuffix
from ..base import (
from ...external.due import BibTeX
@property
def _cmd_prefix(self):
    return '{} '.format(self.inputs.py27_path)