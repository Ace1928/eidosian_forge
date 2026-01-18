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
class DataSinkInputSpec(DynamicTraitedSpec, BaseInterfaceInputSpec):
    """ """
    base_directory = Str(desc='Path to the base directory for storing data.')
    container = Str(desc='Folder within base directory in which to store output')
    parameterization = traits.Bool(True, usedefault=True, desc='store output in parametrized structure')
    strip_dir = Str(desc='path to strip out of filename')
    substitutions = InputMultiPath(traits.Tuple(Str, Str), desc='List of 2-tuples reflecting string to substitute and string to replace it with')
    regexp_substitutions = InputMultiPath(traits.Tuple(Str, Str), desc='List of 2-tuples reflecting a pair of a Python regexp pattern and a replacement string. Invoked after string `substitutions`')
    _outputs = traits.Dict(Str, value={}, usedefault=True)
    remove_dest_dir = traits.Bool(False, usedefault=True, desc='remove dest directory when copying dirs')
    creds_path = Str(desc='Filepath to AWS credentials file for S3 bucket access; if not specified, the credentials will be taken from the AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables')
    encrypt_bucket_keys = traits.Bool(desc='Flag indicating whether to use S3 server-side AES-256 encryption')
    bucket = traits.Any(desc='Boto3 S3 bucket for manual override of bucket')
    local_copy = Str(desc='Copy files locally as well as to S3 bucket')

    def __setattr__(self, key, value):
        if key not in self.copyable_trait_names():
            if not isdefined(value):
                super(DataSinkInputSpec, self).__setattr__(key, value)
            self._outputs[key] = value
        else:
            if key in self._outputs:
                self._outputs[key] = value
            super(DataSinkInputSpec, self).__setattr__(key, value)