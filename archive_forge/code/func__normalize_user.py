import struct
from os_ken.ofproto import ofproto_common
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.lib import type_desc
def _normalize_user(oxx, mod, k, uv):
    try:
        from_user = getattr(mod, oxx + '_from_user')
        n, v, m = from_user(k, uv)
    except:
        return (k, uv)
    if m is not None:
        v = b''.join((struct.Struct('>B').pack(int(x) & int(y)) for x, y in zip(v, m)))
    try:
        to_user = getattr(mod, oxx + '_to_user')
        k2, uv2 = to_user(n, v, m)
    except:
        return (k, uv)
    assert k2 == k
    return (k2, uv2)