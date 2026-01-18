import base64
import logging
import netaddr
from os_ken.lib import dpid
from os_ken.lib import hub
from os_ken.ofproto import ofproto_v1_2
def _reserved_num_from_user(self, num, prefix):
    try:
        return str_to_int(num)
    except ValueError:
        try:
            if num.startswith(prefix):
                return getattr(self.ofproto, num.upper())
            else:
                return getattr(self.ofproto, prefix + num.upper())
        except AttributeError:
            LOG.warning('Cannot convert argument to reserved number: %s', num)
    return num