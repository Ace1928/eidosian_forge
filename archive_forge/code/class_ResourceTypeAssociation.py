import wsme
from wsme import types
from glance.common.wsme_utils import WSMEModelTransformer
class ResourceTypeAssociation(types.Base, WSMEModelTransformer):
    name = wsme.wsattr(types.text, mandatory=True)
    prefix = wsme.wsattr(types.text, mandatory=False)
    properties_target = wsme.wsattr(types.text, mandatory=False)
    created_at = wsme.wsattr(types.text, mandatory=False)
    updated_at = wsme.wsattr(types.text, mandatory=False)

    def __init__(self, **kwargs):
        super(ResourceTypeAssociation, self).__init__(**kwargs)