from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import re
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.endpoints import exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import retry
import six
def ValidateEmailString(email):
    """Returns true if the input is a valid email string.

  This method uses a somewhat rudimentary regular expression to determine
  input validity, but it should suffice for basic sanity checking.

  It also verifies that the email string is no longer than 254 characters,
  since that is the specified maximum length.

  Args:
    email: The email string to validate

  Returns:
    A bool -- True if the input is valid, False otherwise
  """
    return EMAIL_REGEX.match(email or '') is not None and len(email) <= 254