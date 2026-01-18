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
class FSSourceInputSpec(BaseInterfaceInputSpec):
    subjects_dir = Directory(exists=True, mandatory=True, desc='Freesurfer subjects directory.')
    subject_id = Str(mandatory=True, desc='Subject name for whom to retrieve data')
    hemi = traits.Enum('both', 'lh', 'rh', usedefault=True, desc='Selects hemisphere specific outputs')