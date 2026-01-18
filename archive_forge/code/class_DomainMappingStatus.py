from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DomainMappingStatus(_messages.Message):
    """The current state of the Domain Mapping.

  Fields:
    conditions: Array of observed DomainMappingConditions, indicating the
      current state of the DomainMapping.
    mappedRouteName: The name of the route that the mapping currently points
      to.
    observedGeneration: ObservedGeneration is the 'Generation' of the
      DomainMapping that was last processed by the controller. Clients polling
      for completed reconciliation should poll until observedGeneration =
      metadata.generation and the Ready condition's status is True or False.
    resourceRecords: The resource records required to configure this domain
      mapping. These records must be added to the domain's DNS configuration
      in order to serve the application via this domain mapping.
    url: Optional. Not supported by Cloud Run.
  """
    conditions = _messages.MessageField('GoogleCloudRunV1Condition', 1, repeated=True)
    mappedRouteName = _messages.StringField(2)
    observedGeneration = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    resourceRecords = _messages.MessageField('ResourceRecord', 4, repeated=True)
    url = _messages.StringField(5)