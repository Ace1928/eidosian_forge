from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AppengineAppsDomainMappingsPatchRequest(_messages.Message):
    """A AppengineAppsDomainMappingsPatchRequest object.

  Fields:
    domainMapping: A DomainMapping resource to be passed as the request body.
    name: Name of the resource to update. Example:
      apps/myapp/domainMappings/example.com.
    updateMask: Required. Standard field mask for the set of fields to be
      updated.
  """
    domainMapping = _messages.MessageField('DomainMapping', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)