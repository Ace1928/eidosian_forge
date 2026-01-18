from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import datetime
import fnmatch
import json
from googlecloudsdk.command_lib.anthos.config.sync.common import exceptions
from googlecloudsdk.command_lib.anthos.config.sync.common import utils
from googlecloudsdk.core import log
def _GetPathValue(obj, paths, default_value=None):
    """Get the value at the given paths from the input json object.

  Args:
    obj: The json object that represents a RepoSync|RootSync CR.
    paths: [] The string paths in the json object.
    default_value: The default value to return if the path value is not found in
      the object.

  Returns:
    The field value of the given paths if found. Otherwise it returns None.
  """
    if not obj:
        return default_value
    for p in paths:
        if p in obj:
            obj = obj[p]
        else:
            return default_value
    return obj