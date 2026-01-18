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
class DataGrabberInputSpec(DynamicTraitedSpec, BaseInterfaceInputSpec):
    base_directory = Directory(exists=True, desc='Path to the base directory consisting of subject data.')
    raise_on_empty = traits.Bool(True, usedefault=True, desc='Generate exception if list is empty for a given field')
    drop_blank_outputs = traits.Bool(False, usedefault=True, desc='Remove ``None`` entries from output lists')
    sort_filelist = traits.Bool(mandatory=True, desc='Sort the filelist that matches the template')
    template = Str(mandatory=True, desc='Layout used to get files. relative to base directory if defined')
    template_args = traits.Dict(key_trait=Str, value_trait=traits.List(traits.List), desc='Information to plug into template')