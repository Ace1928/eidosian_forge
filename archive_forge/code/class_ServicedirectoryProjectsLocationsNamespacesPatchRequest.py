from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServicedirectoryProjectsLocationsNamespacesPatchRequest(_messages.Message):
    """A ServicedirectoryProjectsLocationsNamespacesPatchRequest object.

  Fields:
    name: Immutable. The resource name for the namespace in the format
      `projects/*/locations/*/namespaces/*`.
    namespace: A Namespace resource to be passed as the request body.
    updateMask: Required. List of fields to be updated in this request.
  """
    name = _messages.StringField(1, required=True)
    namespace = _messages.MessageField('Namespace', 2)
    updateMask = _messages.StringField(3)