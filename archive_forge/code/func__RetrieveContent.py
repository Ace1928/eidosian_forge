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
def _RetrieveContent(self, raise_exceptions=True):
    """Helper function for both Exists and GetContent.

    Args:
      raise_exceptions: Set to false if you just want to know if the file
          actually exists.

    Returns:
      True if we successfully got the content of the URL. Returns False is
      raise_exceptions is False.

    Raises:
      HTTPError: If raise_exceptions is True, will raise exceptions for 4xx or
          5xx response codes instead of returning False.
    """
    r = requests.get(self.full_path)
    try:
        r.raise_for_status()
    except requests.exceptions.HTTPError as e:
        if raise_exceptions:
            raise e
        return False
    self.content = r.text
    return True