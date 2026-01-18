from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import ipaddress
import re
from googlecloudsdk.api_lib.composer import util as api_util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.composer import parsers
from googlecloudsdk.command_lib.composer import util as command_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import properties
import six
def ValidateIpRanges(ip_ranges):
    """Validates list of IP ranges.

  Raises exception when any of the given strings is not a valid IPv4
  or IPv6 network IP range.
  Args:
    ip_ranges: [string], list of IP ranges to validate
  """
    for ip_range in ip_ranges:
        if six.PY2:
            ip_range = ip_range.decode()
        try:
            ipaddress.ip_network(ip_range)
        except:
            raise command_util.InvalidUserInputError('Invalid IP range: [{}].'.format(ip_range))