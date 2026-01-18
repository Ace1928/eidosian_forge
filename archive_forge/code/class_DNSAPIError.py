from __future__ import (absolute_import, division, print_function)
import abc
from ansible.module_utils import six
from ansible_collections.community.dns.plugins.module_utils.zone import (
class DNSAPIError(Exception):
    pass