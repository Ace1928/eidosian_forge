from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core.util import times
import ipaddress
import six
def GetIpVersion(ip_address):
    """Given an ip address, determine IP version.

  Args:
    ip_address: string, IP address to test IP version of

  Returns:
    int, the IP version if it could be determined or IP_VERSION_UNKNOWN
    otherwise.
  """
    try:
        version = ipaddress.ip_address(six.text_type(ip_address)).version
        if version not in (IP_VERSION_4, IP_VERSION_6):
            raise ValueError('Reported IP version not recognized.')
        return version
    except ValueError:
        return IP_VERSION_UNKNOWN