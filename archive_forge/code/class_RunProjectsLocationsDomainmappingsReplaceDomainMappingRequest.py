from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class RunProjectsLocationsDomainmappingsReplaceDomainMappingRequest(_messages.Message):
    """A RunProjectsLocationsDomainmappingsReplaceDomainMappingRequest object.

  Fields:
    domainMapping: A DomainMapping resource to be passed as the request body.
    name: The name of the domain mapping being retrieved. If needed, replace
      {namespace_id} with the project ID.
  """
    domainMapping = _messages.MessageField('DomainMapping', 1)
    name = _messages.StringField(2, required=True)