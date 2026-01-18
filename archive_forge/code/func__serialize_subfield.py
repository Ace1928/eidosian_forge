import struct
from os_ken import utils
from os_ken.lib import type_desc
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import ofproto_common
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto.ofproto_parser import StringifyMixin
@staticmethod
def _serialize_subfield(subfield):
    field, ofs = subfield
    buf = bytearray()
    n = ofp.oxm_from_user_header(field)
    ofp.oxm_serialize_header(n, buf, 0)
    assert len(buf) == 4
    msg_pack_into('!H', buf, 4, ofs)
    return buf