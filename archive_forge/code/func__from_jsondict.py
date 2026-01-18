from os_ken.ofproto.oxx_fields import (
def _from_jsondict(j):
    tlv = j['OXSTlv']
    field = tlv['field']
    value = tlv['value']
    return (field, value)