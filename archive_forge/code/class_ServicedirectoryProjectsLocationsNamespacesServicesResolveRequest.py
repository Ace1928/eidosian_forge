from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServicedirectoryProjectsLocationsNamespacesServicesResolveRequest(_messages.Message):
    """A ServicedirectoryProjectsLocationsNamespacesServicesResolveRequest
  object.

  Fields:
    name: Required. The name of the service to resolve.
    resolveServiceRequest: A ResolveServiceRequest resource to be passed as
      the request body.
  """
    name = _messages.StringField(1, required=True)
    resolveServiceRequest = _messages.MessageField('ResolveServiceRequest', 2)