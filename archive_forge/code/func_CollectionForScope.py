from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
@classmethod
def CollectionForScope(cls, scope):
    if scope == cls.ZONE:
        return 'compute.zones'
    if scope == cls.REGION:
        return 'compute.regions'
    raise exceptions.Error('Expected scope to be ZONE or REGION, got {0!r}'.format(scope))