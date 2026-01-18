import struct
from os_ken.ofproto import ofproto_common
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.lib import type_desc
def _to_user_header(oxx, num_to_field, n):
    name, t = _get_field_info_by_number(oxx, num_to_field, n)
    return name