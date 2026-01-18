from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
class ScopeEnum(enum.Enum):
    """Enum representing GCE scope."""
    ZONE = ('zone', 'a ', properties.VALUES.compute.zone.Get)
    REGION = ('region', 'a ', properties.VALUES.compute.region.Get)
    GLOBAL = ('global', '', lambda: None)

    def __init__(self, flag_name, prefix, property_func):
        self.param_name = flag_name
        self.flag_name = flag_name
        self.prefix = prefix
        self.property_func = property_func

    @classmethod
    def CollectionForScope(cls, scope):
        if scope == cls.ZONE:
            return 'compute.zones'
        if scope == cls.REGION:
            return 'compute.regions'
        raise exceptions.Error('Expected scope to be ZONE or REGION, got {0!r}'.format(scope))