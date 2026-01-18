import collections
import ipaddress
from oslo_utils import uuidutils
import re
import string
from manilaclient import api_versions
from manilaclient import base
from manilaclient.common import constants
from manilaclient import exceptions
from manilaclient.v2 import share_instances
def _validate_access(self, access_type, access, valid_access_types=None, enable_ipv6=False):
    if not valid_access_types:
        valid_access_types = ('ip', 'user', 'cert')
    if access_type in valid_access_types:
        if access_type == 'ip':
            try:
                if enable_ipv6:
                    ipaddress.ip_network(str(access))
                else:
                    ipaddress.IPv4Network(str(access))
            except ValueError as error:
                raise exceptions.CommandError(str(error))
        elif access_type == 'user':
            self._validate_username(access)
        elif access_type == 'cert':
            self._validate_common_name(access.strip())
        elif access_type == 'cephx':
            self._validate_cephx_id(access.strip())
    else:
        msg = 'Only following access types are supported: %s' % ', '.join(valid_access_types)
        raise exceptions.CommandError(msg)