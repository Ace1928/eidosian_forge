from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
import os
import re
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.updater import installers
from googlecloudsdk.core.updater import schemas
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import requests
import six
@staticmethod
def _GetAbsoluteURL(url, value):
    """Convert the potentially relative value into an absolute URL.

    Args:
      url: str, The URL of the component snapshot this value was found in.
      value: str, The value of the field to make absolute.  If it is already an
        absolute URL, it is returned as-is.  If it is relative, it's path
        is assumed to be relative to the component snapshot URL.

    Returns:
      str, The absolute URL.
    """
    if ComponentSnapshot.ABSOLUTE_RE.search(value):
        return value
    return os.path.dirname(url) + '/' + value