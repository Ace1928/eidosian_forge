from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAssetV1Edge(_messages.Message):
    """A directional edge.

  Fields:
    sourceNode: The source node of the edge. For example, it could be a full
      resource name for a resource node or an email of an identity.
    targetNode: The target node of the edge. For example, it could be a full
      resource name for a resource node or an email of an identity.
  """
    sourceNode = _messages.StringField(1)
    targetNode = _messages.StringField(2)