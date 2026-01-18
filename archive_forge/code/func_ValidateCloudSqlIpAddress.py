from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import ipaddress
import re
from typing import Any
from googlecloudsdk.calliope import exceptions
def ValidateCloudSqlIpAddress(ip_address):
    """Validates the Cloud SQL IP address.

  Args:
    ip_address: the Cloud SQL IP address.

  Returns:
    the IP address.
  Raises:
    BadArgumentException: when the IP address is invalid.
  """
    try:
        ipaddress.IPv4Address(ip_address)
        return ip_address
    except ValueError:
        raise exceptions.BadArgumentException('--ip-address', 'Invalid IP address.')