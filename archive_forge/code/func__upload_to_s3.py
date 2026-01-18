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
def _upload_to_s3(self, bucket, src, dst):
    """
        Method to upload outputs to S3 bucket instead of on local disk
        """
    import hashlib
    import os
    from botocore.exceptions import ClientError
    s3_str = 's3://'
    s3_prefix = s3_str + bucket.name
    if dst.lower().startswith(s3_str):
        dst = s3_str + dst[len(s3_str):]
    if os.path.isdir(src):
        src_files = []
        for root, dirs, files in os.walk(src):
            src_files.extend([os.path.join(root, fil) for fil in files])
        dst_files = [os.path.join(dst, src_f.split(src)[1]) for src_f in src_files]
    else:
        src_files = [src]
        dst_files = [dst]
    for src_idx, src_f in enumerate(src_files):
        dst_f = dst_files[src_idx]
        dst_k = dst_f.replace(s3_prefix, '').lstrip('/')
        try:
            dst_obj = bucket.Object(key=dst_k)
            dst_md5 = dst_obj.e_tag.strip('"')
            src_read = open(src_f, 'rb').read()
            src_md5 = hashlib.md5(src_read).hexdigest()
            if dst_md5 == src_md5:
                iflogger.info('File %s already exists on S3, skipping...', dst_f)
                continue
            else:
                iflogger.info('Overwriting previous S3 file...')
        except ClientError:
            iflogger.info('New file to S3')
        iflogger.info('Uploading %s to S3 bucket, %s, as %s...', src_f, bucket.name, dst_f)
        if self.inputs.encrypt_bucket_keys:
            extra_args = {'ServerSideEncryption': 'AES256'}
        else:
            extra_args = {}
        bucket.upload_file(src_f, dst_k, ExtraArgs=extra_args, Callback=ProgressPercentage(src_f))