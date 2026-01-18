from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.tasks import app
from googlecloudsdk.command_lib.tasks import constants
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import http_encoding
def ParseAttributes(attributes):
    """Parse "--attributes" flag as an argparse type.

  The flag is given as a Calliope ArgDict:

      --attributes key1=value1,key2=value2

  Args:
    attributes: str, the value of the --attributes flag.

  Returns:
    dict, a dict with 'additionalProperties' as a key, and a list of dicts
        containing key-value pairs as the value.
  """
    attributes = arg_parsers.ArgDict()(attributes)
    return {'additionalProperties': [{'key': key, 'value': value} for key, value in sorted(attributes.items())]}