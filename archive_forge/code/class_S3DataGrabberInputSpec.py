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
class S3DataGrabberInputSpec(DynamicTraitedSpec, BaseInterfaceInputSpec):
    anon = traits.Bool(False, usedefault=True, desc='Use anonymous connection to s3.  If this is set to True, boto may print a urlopen error, but this does not prevent data from being downloaded.')
    region = Str('us-east-1', usedefault=True, desc='Region of s3 bucket')
    bucket = Str(mandatory=True, desc='Amazon S3 bucket where your data is stored')
    bucket_path = Str('', usedefault=True, desc='Location within your bucket for subject data.')
    local_directory = Directory(exists=True, desc='Path to the local directory for subject data to be downloaded and accessed. Should be on HDFS for Spark jobs.')
    raise_on_empty = traits.Bool(True, usedefault=True, desc='Generate exception if list is empty for a given field')
    sort_filelist = traits.Bool(mandatory=True, desc='Sort the filelist that matches the template')
    template = Str(mandatory=True, desc='Layout used to get files. Relative to bucket_path if defined.Uses regex rather than glob style formatting.')
    template_args = traits.Dict(key_trait=Str, value_trait=traits.List(traits.List), desc='Information to plug into template')