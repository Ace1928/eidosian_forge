import wsme
from wsme import types
class ItemType(types.Base):
    type = wsme.wsattr(types.text, mandatory=True)
    enum = wsme.wsattr([types.text], mandatory=False)
    _wsme_attr_order = ('type', 'enum')

    def __init__(self, **kwargs):
        super(ItemType, self).__init__(**kwargs)