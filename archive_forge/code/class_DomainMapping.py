from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DomainMapping(_messages.Message):
    """Resource to hold the state and status of a user's domain mapping. NOTE:
  This resource is currently in Beta.

  Fields:
    apiVersion: The API version for this call such as
      "domains.cloudrun.com/v1".
    kind: The kind of resource, in this case "DomainMapping".
    metadata: Metadata associated with this BuildTemplate.
    spec: The spec for this DomainMapping.
    status: The current status of the DomainMapping.
  """
    apiVersion = _messages.StringField(1)
    kind = _messages.StringField(2)
    metadata = _messages.MessageField('ObjectMeta', 3)
    spec = _messages.MessageField('DomainMappingSpec', 4)
    status = _messages.MessageField('DomainMappingStatus', 5)