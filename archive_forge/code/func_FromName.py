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
@classmethod
def FromName(cls, bucket, name):
    """Create an object reference after ensuring the name is valid."""
    _ValidateBucketName(bucket)
    if not 0 <= len(name.encode('utf-8')) <= 1024:
        raise InvalidObjectNameError(name, VALID_OBJECT_LENGTH_MESSAGE)
    if '\r' in name or '\n' in name:
        raise InvalidObjectNameError(name, VALID_OBJECT_CHARS_MESSAGE)
    return cls(bucket, name)