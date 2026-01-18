from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RunProjectsLocationsDomainmappingsCreateRequest(_messages.Message):
    """A RunProjectsLocationsDomainmappingsCreateRequest object.

  Fields:
    domainMapping: A DomainMapping resource to be passed as the request body.
    dryRun: Indicates that the server should validate the request and populate
      default values without persisting the request. Supported values: `all`
    parent: Required. The namespace in which the domain mapping should be
      created. For Cloud Run (fully managed), replace {namespace} with the
      project ID or number. It takes the form namespaces/{namespace}. For
      example: namespaces/PROJECT_ID
  """
    domainMapping = _messages.MessageField('DomainMapping', 1)
    dryRun = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)