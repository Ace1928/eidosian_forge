from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TypeMeta(_messages.Message):
    """TypeMeta is the type information needed for content unmarshalling of
  Kubernetes resources in the manifest.

  Fields:
    apiVersion: APIVersion of the resource (e.g. v1).
    kind: Kind of the resource (e.g. Deployment).
  """
    apiVersion = _messages.StringField(1)
    kind = _messages.StringField(2)