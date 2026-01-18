import wsme
from wsme import types
from glance.api.v2.model.metadef_property_item_type import ItemType
from glance.common.wsme_utils import WSMEModelTransformer
class PropertyType(types.Base, WSMEModelTransformer):
    name = wsme.wsattr(types.text, mandatory=False)
    type = wsme.wsattr(types.text, mandatory=True)
    title = wsme.wsattr(types.text, mandatory=True)
    description = wsme.wsattr(types.text, mandatory=False)
    operators = wsme.wsattr([types.text], mandatory=False)
    default = wsme.wsattr(types.bytes, mandatory=False)
    readonly = wsme.wsattr(bool, mandatory=False)
    minimum = wsme.wsattr(int, mandatory=False)
    maximum = wsme.wsattr(int, mandatory=False)
    enum = wsme.wsattr([types.text], mandatory=False)
    pattern = wsme.wsattr(types.text, mandatory=False)
    minLength = wsme.wsattr(int, mandatory=False)
    maxLength = wsme.wsattr(int, mandatory=False)
    confidential = wsme.wsattr(bool, mandatory=False)
    items = wsme.wsattr(ItemType, mandatory=False)
    uniqueItems = wsme.wsattr(bool, mandatory=False)
    minItems = wsme.wsattr(int, mandatory=False)
    maxItems = wsme.wsattr(int, mandatory=False)
    additionalItems = wsme.wsattr(bool, mandatory=False)

    def __init__(self, **kwargs):
        super(PropertyType, self).__init__(**kwargs)