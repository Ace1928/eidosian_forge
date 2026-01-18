from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import atexit
import json
import os
from boto import config
import gslib
from gslib import exception
from gslib.utils import boto_util
from gslib.utils import execution_util
def _read_metadata_file(metadata_path):
    """Loads context aware metadata from the given path.

  Returns:
      dict: The metadata JSON.

  Raises:
      CertProvisionError: If failed to parse metadata as JSON.
  """
    try:
        with open(metadata_path) as f:
            return json.load(f)
    except ValueError as e:
        raise CertProvisionError(e)