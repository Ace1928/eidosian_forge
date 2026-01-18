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
class SelectFilesInputSpec(DynamicTraitedSpec, BaseInterfaceInputSpec):
    base_directory = Directory(exists=True, desc='Root path common to templates.')
    sort_filelist = traits.Bool(True, usedefault=True, desc='When matching multiple files, return them in sorted order.')
    raise_on_empty = traits.Bool(True, usedefault=True, desc='Raise an exception if a template pattern matches no files.')
    force_lists = traits.Either(traits.Bool(), traits.List(Str()), default=False, usedefault=True, desc='Whether to return outputs as a list even when only one file matches the template. Either a boolean that applies to all output fields or a list of output field names to coerce to a list')