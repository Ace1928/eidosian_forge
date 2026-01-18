from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkehubProjectsLocationsScopesNamespacesResourcequotasPatchRequest(_messages.Message):
    """A GkehubProjectsLocationsScopesNamespacesResourcequotasPatchRequest
  object.

  Fields:
    name: The resource name for the resourcequota itself `projects/{project}/l
      ocations/{location}/scopes/{scope}/namespaces/{namespace}/resourcequotas
      /{resourcequota}`
    resourceQuota: A ResourceQuota resource to be passed as the request body.
    updateMask: Required. The fields to be updated.
  """
    name = _messages.StringField(1, required=True)
    resourceQuota = _messages.MessageField('ResourceQuota', 2)
    updateMask = _messages.StringField(3)