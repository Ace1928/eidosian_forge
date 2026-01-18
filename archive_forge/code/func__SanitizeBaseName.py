from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import glob
import os
import posixpath
import re
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.deployment_manager import exceptions
from googlecloudsdk.api_lib.util import exceptions as api_exceptions
from googlecloudsdk.core import yaml
import googlecloudsdk.core.properties
from googlecloudsdk.core.util import files
import requests
import six
import six.moves.urllib.parse
def _SanitizeBaseName(base_name):
    """Make sure the base_name will be a valid resource name.

  Args:
    base_name: Name of a template file, and therefore not empty.

  Returns:
    base_name with periods and underscores removed,
        and the first letter lowercased.
  """
    sanitized = base_name.replace('.', '-').replace('_', '-')
    return sanitized[0].lower() + sanitized[1:]