import json
import logging
import os
import re
import subprocess
from googlecloudsdk.third_party.appengine._internal import six_subset
class _ContextType(object):
    """Ordered enumeration of context types.

  The ordering is based on which context information will provide the best
  user experience. Higher numbers are considered better than lower numbers.
  Google repositories have the highest ranking because they do not require
  additional authorization to view.
  """
    OTHER = 0
    GIT_UNKNOWN = 1
    GIT_KNOWN_HOST_SSH = 2
    GIT_KNOWN_HOST = 3
    CLOUD_REPO = 4
    SOURCE_CAPTURE = 5