import argparse
import arg_parsers
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import collections
import copy
import decimal
import json
import re
from dateutil import tz
from googlecloudsdk.calliope import arg_parsers_usage_text as usage_text
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
from six.moves import zip  # pylint: disable=redefined-builtin
class HostPort(object):
    """A class for holding host and port information."""
    IPV4_OR_HOST_PATTERN = '^(?P<address>[\\w\\d\\.-]+)?(:|:(?P<port>[\\d]+))?$'
    IPV6_PATTERN = '^(\\[(?P<address>[\\w\\d:]+)\\])(:|:(?P<port>[\\d]+))?$'

    def __init__(self, host, port):
        self.host = host
        self.port = port

    @staticmethod
    def Parse(s, ipv6_enabled=False):
        """Parse the given string into a HostPort object.

    This can be used as an argparse type.

    Args:
      s: str, The string to parse. If ipv6_enabled and host is an IPv6 address,
      it should be placed in square brackets: e.g.
        [2001:db8:0:0:0:ff00:42:8329] or
        [2001:db8:0:0:0:ff00:42:8329]:8080
      ipv6_enabled: boolean, If True then accept IPv6 addresses.

    Raises:
      ArgumentTypeError: If the string is not valid.

    Returns:
      HostPort, The parsed object.
    """
        if not s:
            return HostPort(None, None)
        match = re.match(HostPort.IPV4_OR_HOST_PATTERN, s, re.UNICODE)
        if ipv6_enabled and (not match):
            match = re.match(HostPort.IPV6_PATTERN, s, re.UNICODE)
            if not match:
                raise ArgumentTypeError(_GenerateErrorMessage('Failed to parse host and port. Expected format \n\n  IPv4_ADDRESS_OR_HOSTNAME:PORT\n\nor\n\n  [IPv6_ADDRESS]:PORT\n\n(where :PORT is optional).', user_input=s))
        elif not match:
            raise ArgumentTypeError(_GenerateErrorMessage('Failed to parse host and port. Expected format \n\n  IPv4_ADDRESS_OR_HOSTNAME:PORT\n\n(where :PORT is optional).', user_input=s))
        return HostPort(match.group('address'), match.group('port'))