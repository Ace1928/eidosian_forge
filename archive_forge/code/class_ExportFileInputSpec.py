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
class ExportFileInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='Input file name')
    out_file = File(mandatory=True, desc='Output file name')
    check_extension = traits.Bool(True, usedefault=True, desc='Ensure that the input and output file extensions match')
    clobber = traits.Bool(desc='Permit overwriting existing files')