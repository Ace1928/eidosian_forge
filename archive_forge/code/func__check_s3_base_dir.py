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
def _check_s3_base_dir(self):
    """
        Method to see if the datasink's base directory specifies an
        S3 bucket path; if it does, it parses the path for the bucket
        name in the form 's3://bucket_name/...' and returns it

        Parameters
        ----------

        Returns
        -------
        s3_flag : boolean
            flag indicating whether the base_directory contained an
            S3 bucket path
        bucket_name : string
            name of the S3 bucket to connect to; if the base directory
            is not a valid S3 path, defaults to '<N/A>'
        """
    s3_str = 's3://'
    bucket_name = '<N/A>'
    base_directory = self.inputs.base_directory
    if not isdefined(base_directory):
        s3_flag = False
        return (s3_flag, bucket_name)
    s3_flag = base_directory.lower().startswith(s3_str)
    if s3_flag:
        bucket_name = base_directory[len(s3_str):].partition('/')[0]
    return (s3_flag, bucket_name)