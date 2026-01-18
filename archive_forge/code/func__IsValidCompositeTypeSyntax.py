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
def _IsValidCompositeTypeSyntax(composite_type_name):
    """Returns true if the resource_handle matches composite type syntax.

  Args:
    composite_type_name: a string of the name of the composite type.

  Catches most syntax errors by checking that the string contains the substring
  '/composite:' preceded and followed by at least one character, none of which
  are colons, slashes, or whitespace. Periods may follow the substring, but not
  proceed it.
  """
    return re.match('^[^/:.\\s]+/composite:[^/:\\s]+$', composite_type_name)