from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class NodeMapValue(_messages.Message):
    """Map between node.id and cel node Node id: Expr.id
    (google/api/expr/syntax.proto)

    Messages:
      AdditionalProperty: An additional property for a NodeMapValue object.

    Fields:
      additionalProperties: Additional properties of type NodeMapValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a NodeMapValue object.

      Fields:
        key: Name of the additional property.
        value: A IdentityCaaIntelFrontendCelNode attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('IdentityCaaIntelFrontendCelNode', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)