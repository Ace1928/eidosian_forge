from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import ipaddress
import re
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.core import resources
def IPV4Argument(value):
    """Argparse argument type that checks for a valid ipv4 address."""
    if not IsValidIPV4(value):
        raise arg_parsers.ArgumentTypeError("invalid ipv4 value: '{0}'".format(value))
    return value