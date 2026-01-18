import struct
from os_ken.ofproto import ofproto_common
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.lib import type_desc
def _serialize_header(oxx, mod, n, buf, offset):
    try:
        get_desc = getattr(mod, '_' + oxx + '_field_desc')
        desc = get_desc(n)
        value_len = desc.type.size
    except KeyError:
        value_len = 0
    n, exp_hdr = _make_exp_hdr(oxx, mod, n)
    exp_hdr_len = len(exp_hdr)
    pack_str = '!I%ds' % (exp_hdr_len,)
    msg_pack_into(pack_str, buf, offset, n << 9 | 0 << 8 | exp_hdr_len + value_len, bytes(exp_hdr))
    return struct.calcsize(pack_str)