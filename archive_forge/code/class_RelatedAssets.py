from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RelatedAssets(_messages.Message):
    """DEPRECATED. This message only presents for the purpose of backward-
  compatibility. The server will never populate this message in responses. The
  detailed related assets with the `relationship_type`.

  Fields:
    assets: The peer resources of the relationship.
    relationshipAttributes: The detailed relationship attributes.
  """
    assets = _messages.MessageField('RelatedAsset', 1, repeated=True)
    relationshipAttributes = _messages.MessageField('RelationshipAttributes', 2)