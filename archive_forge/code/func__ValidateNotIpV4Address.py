from __future__ import absolute_import
from __future__ import unicode_literals
import os
import re
from googlecloudsdk.third_party.appengine._internal import six_subset
def _ValidateNotIpV4Address(host):
    """Validate host is not an IPV4 address."""
    matcher = _URL_IP_V4_ADDR_RE.match(host)
    if matcher and sum((1 for x in matcher.groups() if int(x) <= 255)) == 4:
        raise validation.ValidationError("Host may not match an ipv4 address '%s'" % host)
    return matcher