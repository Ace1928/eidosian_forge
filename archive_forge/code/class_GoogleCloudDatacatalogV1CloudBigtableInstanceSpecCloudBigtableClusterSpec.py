from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1CloudBigtableInstanceSpecCloudBigtableClusterSpec(_messages.Message):
    """Spec that applies to clusters of an Instance of Cloud Bigtable.

  Fields:
    displayName: Name of the cluster.
    linkedResource: A link back to the parent resource, in this case Instance.
    location: Location of the cluster, typically a Cloud zone.
    type: Type of the resource. For a cluster this would be "CLUSTER".
  """
    displayName = _messages.StringField(1)
    linkedResource = _messages.StringField(2)
    location = _messages.StringField(3)
    type = _messages.StringField(4)