import struct
from os_ken.ofproto import ofproto_common
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.lib import type_desc
def _get_field_info_by_number(oxx, num_to_field, n):
    try:
        f = num_to_field[n]
        t = f.type
        name = f.name
    except KeyError:
        t = type_desc.UnknownType
        if isinstance(n, int):
            name = 'field_%d' % (n,)
        else:
            raise KeyError('unknown %s field number: %s' % (oxx.upper(), n))
    return (name, t)