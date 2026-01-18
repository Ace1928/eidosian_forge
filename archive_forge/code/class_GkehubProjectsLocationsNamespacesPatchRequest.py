from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkehubProjectsLocationsNamespacesPatchRequest(_messages.Message):
    """A GkehubProjectsLocationsNamespacesPatchRequest object.

  Fields:
    name: The resource name for the namespace
      `projects/{project}/locations/{location}/namespaces/{namespace}`
    namespace: A Namespace resource to be passed as the request body.
    updateMask: Required. The fields to be updated.
  """
    name = _messages.StringField(1, required=True)
    namespace = _messages.MessageField('Namespace', 2)
    updateMask = _messages.StringField(3)