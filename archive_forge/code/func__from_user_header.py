import struct
from os_ken.ofproto import ofproto_common
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.lib import type_desc
def _from_user_header(oxx, name_to_field, name):
    num, t = _get_field_info_by_name(oxx, name_to_field, name)
    return num