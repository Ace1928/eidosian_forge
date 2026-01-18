from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAssetV1p7beta1RelatedAssets(_messages.Message):
    """The detailed related assets with the `relationship_type`.

  Fields:
    assets: The peer resources of the relationship.
    relationshipAttributes: The detailed relation attributes.
  """
    assets = _messages.MessageField('GoogleCloudAssetV1p7beta1RelatedAsset', 1, repeated=True)
    relationshipAttributes = _messages.MessageField('GoogleCloudAssetV1p7beta1RelationshipAttributes', 2)