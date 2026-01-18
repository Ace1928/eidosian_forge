import struct
from os_ken import exception
from os_ken.lib import mac
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto import ether
from os_ken.ofproto import ofproto_parser
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import inet
import logging
def _register_make(cls):
    """class decorator to Register mf make"""
    assert cls.nxm_headers is not None
    assert cls.nxm_headers is not []
    for nxm_header in cls.nxm_headers:
        assert nxm_header not in _MF_FIELDS
        _MF_FIELDS[nxm_header] = cls.make
    return cls