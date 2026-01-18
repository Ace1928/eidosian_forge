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
def _get_head_bucket(s3_resource, bucket_name):
    """Try to get the header info of a bucket, in order to
    check if it exists and its permissions
    """
    import botocore
    try:
        s3_resource.meta.client.head_bucket(Bucket=bucket_name)
    except botocore.exceptions.ClientError as exc:
        error_code = int(exc.response['Error']['Code'])
        if error_code == 403:
            err_msg = 'Access to bucket: %s is denied; check credentials' % bucket_name
            raise Exception(err_msg)
        elif error_code == 404:
            err_msg = 'Bucket: %s does not exist; check spelling and try again' % bucket_name
            raise Exception(err_msg)
        else:
            err_msg = 'Unable to connect to bucket: %s. Error message:\n%s' % (bucket_name, exc)
    except Exception as exc:
        err_msg = 'Unable to connect to bucket: %s. Error message:\n%s' % (bucket_name, exc)
        raise Exception(err_msg)