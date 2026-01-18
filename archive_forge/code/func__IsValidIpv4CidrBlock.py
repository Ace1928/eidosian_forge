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
def _IsValidIpv4CidrBlock(ipv4_cidr_block):
    """Validates that IPV4 CIDR block arg has valid format.

  Intended to be used as an argparse validator.

  Args:
    ipv4_cidr_block: str, the IPV4 CIDR block string to validate

  Returns:
    bool, True if and only if the IPV4 CIDR block is valid
  """
    return ipaddress.IPv4Network(ipv4_cidr_block) is not None