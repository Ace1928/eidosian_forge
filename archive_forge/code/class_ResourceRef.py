from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResourceRef(_messages.Message):
    """Reference to a K8S resource.

  Fields:
    groupKind: GK is the GroupKind of the K8S resource. This field may be
      empty for errors that are not associated with a specific resource.
    name: The name of the K8S resource.
    resourceNamespace: The namespace of the K8S resource. This field may be
      empty for errors that are associated with a cluster-scoped resource.
      Called resource_namespace because namespace is a C++ keyword.
  """
    groupKind = _messages.MessageField('GroupKind', 1)
    name = _messages.StringField(2)
    resourceNamespace = _messages.StringField(3)