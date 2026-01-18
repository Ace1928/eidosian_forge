from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AllocationSpecificSKUReservation(_messages.Message):
    """This reservation type allows to pre allocate specific instance
  configuration. Next ID: 6

  Fields:
    assuredCount: [Output Only] Indicates how many instances are actually
      usable currently.
    count: Specifies the number of resources that are allocated.
    inUseCount: [Output Only] Indicates how many instances are in use.
    instanceProperties: The instance properties for the reservation.
    sourceInstanceTemplate: Specifies the instance template to create the
      reservation. If you use this field, you must exclude the
      instanceProperties field. This field is optional, and it can be a full
      or partial URL. For example, the following are all valid URLs to an
      instance template: -
      https://www.googleapis.com/compute/v1/projects/project
      /global/instanceTemplates/instanceTemplate -
      projects/project/global/instanceTemplates/instanceTemplate -
      global/instanceTemplates/instanceTemplate
  """
    assuredCount = _messages.IntegerField(1)
    count = _messages.IntegerField(2)
    inUseCount = _messages.IntegerField(3)
    instanceProperties = _messages.MessageField('AllocationSpecificSKUAllocationReservedInstanceProperties', 4)
    sourceInstanceTemplate = _messages.StringField(5)