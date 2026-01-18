from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CompositeType(_messages.Message):
    """Holds the composite type.

  Enums:
    StatusValueValuesEnum:

  Fields:
    description: An optional textual description of the resource; provided by
      the client when the resource is created.
    id: A string attribute.
    insertTime: Output only. Creation timestamp in RFC3339 text format.
    labels: Map of labels; provided by the client when the resource is created
      or updated. Specifically: Label keys must be between 1 and 63 characters
      long and must conform to the following regular expression:
      `[a-z]([-a-z0-9]*[a-z0-9])?` Label values must be between 0 and 63
      characters long and must conform to the regular expression
      `([a-z]([-a-z0-9]*[a-z0-9])?)?`.
    name: Name of the composite type, must follow the expression:
      `[a-z]([-a-z0-9_.]{0,61}[a-z0-9])?`.
    operation: Output only. The Operation that most recently ran, or is
      currently running, on this composite type.
    selfLink: Output only. Server defined URL for the resource.
    status: A StatusValueValuesEnum attribute.
    templateContents: Files for the template type.
  """

    class StatusValueValuesEnum(_messages.Enum):
        """StatusValueValuesEnum enum type.

    Values:
      UNKNOWN_STATUS: <no description>
      DEPRECATED: <no description>
      EXPERIMENTAL: <no description>
      SUPPORTED: <no description>
    """
        UNKNOWN_STATUS = 0
        DEPRECATED = 1
        EXPERIMENTAL = 2
        SUPPORTED = 3
    description = _messages.StringField(1)
    id = _messages.IntegerField(2, variant=_messages.Variant.UINT64)
    insertTime = _messages.StringField(3)
    labels = _messages.MessageField('CompositeTypeLabelEntry', 4, repeated=True)
    name = _messages.StringField(5)
    operation = _messages.MessageField('Operation', 6)
    selfLink = _messages.StringField(7)
    status = _messages.EnumField('StatusValueValuesEnum', 8)
    templateContents = _messages.MessageField('TemplateContents', 9)