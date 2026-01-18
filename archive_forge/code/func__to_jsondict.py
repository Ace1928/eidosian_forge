from os_ken.ofproto.oxx_fields import (
def _to_jsondict(k, uv):
    return {'OXSTlv': {'field': k, 'value': uv}}