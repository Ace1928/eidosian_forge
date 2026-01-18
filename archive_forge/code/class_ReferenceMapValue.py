from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class ReferenceMapValue(_messages.Message):
    """A map from expression ids to resolved references. The following
    entries are in this table: - An Ident or Select expression is represented
    here if it resolves to a declaration. For instance, if `a.b.c` is
    represented by `select(select(id(a), b), c)`, and `a.b` resolves to a
    declaration, while `c` is a field selection, then the reference is
    attached to the nested select expression (but not to the id or or the
    outer select). In turn, if `a` resolves to a declaration and `b.c` are
    field selections, the reference is attached to the ident expression. -
    Every Call expression has an entry here, identifying the function being
    called. - Every CreateStruct expression for a message has an entry,
    identifying the message.

    Messages:
      AdditionalProperty: An additional property for a ReferenceMapValue
        object.

    Fields:
      additionalProperties: Additional properties of type ReferenceMapValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a ReferenceMapValue object.

      Fields:
        key: Name of the additional property.
        value: A GoogleApiExprReference attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('GoogleApiExprReference', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)