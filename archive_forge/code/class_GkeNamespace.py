from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkeNamespace(_messages.Message):
    """GKE Namespace. The field names correspond to the resource metadata
  labels on monitored resources that fall under a namespace (for example,
  k8s_container or k8s_pod).

  Fields:
    clusterName: The name of the parent cluster.
    location: The location of the parent cluster. This may be a zone or
      region.
    namespaceName: The name of this namespace.
    projectId: Output only. The project this resource lives in. For legacy
      services migrated from the Custom type, this may be a distinct project
      from the one parenting the service itself.
  """
    clusterName = _messages.StringField(1)
    location = _messages.StringField(2)
    namespaceName = _messages.StringField(3)
    projectId = _messages.StringField(4)