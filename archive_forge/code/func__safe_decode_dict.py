from oslo_utils import encodeutils
from neutronclient._i18n import _
def _safe_decode_dict(kwargs):
    for k, v in kwargs.items():
        kwargs[k] = encodeutils.safe_decode(v)
    return kwargs