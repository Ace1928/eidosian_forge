import wsme
from wsme import types
from glance.api.v2.model.metadef_property_item_type import ItemType
from glance.common.wsme_utils import WSMEModelTransformer
class PropertyTypes(types.Base, WSMEModelTransformer):
    properties = wsme.wsattr({types.text: PropertyType}, mandatory=False)

    def __init__(self, **kwargs):
        super(PropertyTypes, self).__init__(**kwargs)