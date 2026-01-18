from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import random
import re
import string
import time
from typing import Dict, Optional
from apitools.base.py import exceptions as http_exceptions
from apitools.base.py import http_wrapper
from apitools.base.py import transfer
from apitools.base.py import util as http_util
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.functions import exceptions
from googlecloudsdk.command_lib.util import gcloudignore
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core import transports
from googlecloudsdk.core.util import archive
from googlecloudsdk.core.util import files as file_utils
import six
from six.moves import http_client
from six.moves import range
def _ValidateDirectoryExistsOrRaise(directory: str) -> None:
    """Validates that the given directory exists.

  Args:
    directory: a local path to the directory provided by user.

  Returns:
    The argument provided, if found valid.
  Raises:
    SourceArgumentError: If the user provided an invalid directory.
  """
    if not os.path.exists(directory):
        raise exceptions.SourceArgumentError('Provided directory does not exist')
    if not os.path.isdir(directory):
        raise exceptions.SourceArgumentError('Provided path does not point to a directory')