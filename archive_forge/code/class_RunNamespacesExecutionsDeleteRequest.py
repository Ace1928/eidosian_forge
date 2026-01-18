from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RunNamespacesExecutionsDeleteRequest(_messages.Message):
    """A RunNamespacesExecutionsDeleteRequest object.

  Fields:
    apiVersion: Optional. Cloud Run currently ignores this parameter.
    kind: Optional. Cloud Run currently ignores this parameter.
    name: Required. The name of the execution to delete. Replace {namespace}
      with the project ID or number. It takes the form namespaces/{namespace}.
      For example: namespaces/PROJECT_ID
    propagationPolicy: Optional. Specifies the propagation policy of delete.
      Cloud Run currently ignores this setting.
  """
    apiVersion = _messages.StringField(1)
    kind = _messages.StringField(2)
    name = _messages.StringField(3, required=True)
    propagationPolicy = _messages.StringField(4)