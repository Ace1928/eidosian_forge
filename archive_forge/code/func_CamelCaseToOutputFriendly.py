from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import io
import re
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.resource import resource_printer
import ipaddr
import six
def CamelCaseToOutputFriendly(string):
    """Converts camel case text into output friendly text.

  Args:
    string: The string to convert.

  Returns:
    The string converted from CamelCase to output friendly text.

  Examples:
    'camelCase' -> 'camel case'
    'CamelCase' -> 'camel case'
    'camelTLA' -> 'camel tla'
  """
    return re.sub('([A-Z]+)', ' \\1', string).strip().lower()