from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RunNamespacesRevisionsDeleteRequest(_messages.Message):
    """A RunNamespacesRevisionsDeleteRequest object.

  Fields:
    apiVersion: Cloud Run currently ignores this parameter.
    dryRun: Indicates that the server should validate the request and populate
      default values without persisting the request. Supported values: `all`
    kind: Cloud Run currently ignores this parameter.
    name: The name of the revision to delete. For Cloud Run (fully managed),
      replace {namespace} with the project ID or number. It takes the form
      namespaces/{namespace}. For example: namespaces/PROJECT_ID
    propagationPolicy: Specifies the propagation policy of delete. Cloud Run
      currently ignores this setting, and deletes in the background.
  """
    apiVersion = _messages.StringField(1)
    dryRun = _messages.StringField(2)
    kind = _messages.StringField(3)
    name = _messages.StringField(4, required=True)
    propagationPolicy = _messages.StringField(5)