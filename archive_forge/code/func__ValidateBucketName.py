from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import os
import re
import string
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.command_lib.util import gcloudignore
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import files as file_utils
from googlecloudsdk.core.util import platforms
import six
def _ValidateBucketName(name):
    """Validate the given bucket name according to the naming requirements.

  See https://cloud.google.com/storage/docs/naming#requirements

  Args:
    name: the name of the bucket, not including 'gs://'

  Raises:
    InvalidBucketNameError: if the given bucket name is invalid
  """
    components = name.split('.')
    if not 3 <= len(name) <= 222 or any((len(c) > 63 for c in components)):
        raise InvalidBucketNameError(name, VALID_BUCKET_LENGTH_MESSAGE)
    if set(name) - set(string.ascii_lowercase + string.digits + '-_.'):
        raise InvalidBucketNameError(name, VALID_BUCKET_CHARS_MESSAGE)
    if set(name[0] + name[-1]) - set(string.ascii_lowercase + string.digits):
        raise InvalidBucketNameError(name, VALID_BUCKET_START_END_MESSAGE)
    if len(components) == 4 and ''.join(components).isdigit():
        raise InvalidBucketNameError(name, VALID_BUCKET_DOTTED_DECIMAL_MESSAGE)