import base64
import logging
import netaddr
from os_ken.lib import dpid
from os_ken.lib import hub
from os_ken.ofproto import ofproto_v1_2
def _reserved_num_to_user(self, num, prefix):
    for k, v in self.ofproto.__dict__.items():
        if k not in self.deprecated_value and k.startswith(prefix) and (v == num):
            return k.replace(prefix, '')
    return num