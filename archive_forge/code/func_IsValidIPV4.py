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
def IsValidIPV4(ip):
    """Accepts an ipv4 address in string form and returns True if valid."""
    match = re.match('^(\\d{1,3})\\.(\\d{1,3})\\.(\\d{1,3})\\.(\\d{1,3})$', ip)
    if not match:
        return False
    octets = [int(x) for x in match.groups()]
    if octets[0] == 0:
        return False
    for n in octets:
        if n < 0 or n > 255:
            return False
    return True