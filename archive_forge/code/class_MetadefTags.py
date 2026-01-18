import wsme
from wsme import types
from glance.common import wsme_utils
class MetadefTags(types.Base, wsme_utils.WSMEModelTransformer):
    tags = wsme.wsattr([MetadefTag], mandatory=False)