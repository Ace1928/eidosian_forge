import wsme
from wsme import types
from glance.common.wsme_utils import WSMEModelTransformer
class ResourceTypes(types.Base, WSMEModelTransformer):
    resource_types = wsme.wsattr([ResourceType], mandatory=False)

    def __init__(self, **kwargs):
        super(ResourceTypes, self).__init__(**kwargs)