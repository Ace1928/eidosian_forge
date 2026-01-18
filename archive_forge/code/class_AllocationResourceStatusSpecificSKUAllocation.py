from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AllocationResourceStatusSpecificSKUAllocation(_messages.Message):
    """Contains Properties set for the reservation.

  Fields:
    sourceInstanceTemplateId: ID of the instance template used to populate
      reservation properties.
  """
    sourceInstanceTemplateId = _messages.StringField(1)