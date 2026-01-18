from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import collections
import enum
import os
import re
import subprocess
from boto import config
from gslib import exception
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.utils import boto_util
from gslib.utils import constants
from gslib.utils import system_util
def _get_validated_gcloud_binary_path():
    gcloud_binary_path = os.environ.get('GCLOUD_BINARY_PATH')
    if gcloud_binary_path:
        return gcloud_binary_path
    cloudsdk_root = os.environ.get('CLOUDSDK_ROOT_DIR')
    if cloudsdk_root is None:
        raise exception.GcloudStorageTranslationError('Requested to use "gcloud storage" but the gcloud binary path cannot be found. This might happen if you attempt to use gsutil that was not installed via Cloud SDK. You can manually set the `CLOUDSDK_ROOT_DIR` environment variable to point to the google-cloud-sdk installation directory to resolve the issue. Alternatively, you can set `use_gcloud_storage=False` to disable running the command using gcloud storage.')
    return _get_gcloud_binary_path(cloudsdk_root)