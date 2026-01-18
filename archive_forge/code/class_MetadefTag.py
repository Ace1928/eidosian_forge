import wsme
from wsme import types
from glance.common import wsme_utils
class MetadefTag(types.Base, wsme_utils.WSMEModelTransformer):
    name = wsme.wsattr(types.text, mandatory=True)
    created_at = wsme.wsattr(types.text, mandatory=False)
    updated_at = wsme.wsattr(types.text, mandatory=False)