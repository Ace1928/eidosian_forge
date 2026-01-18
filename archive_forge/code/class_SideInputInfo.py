from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SideInputInfo(_messages.Message):
    """Information about a side input of a DoFn or an input of a SeqDoFn.

  Messages:
    KindValue: How to interpret the source element(s) as a side input value.

  Fields:
    kind: How to interpret the source element(s) as a side input value.
    sources: The source(s) to read element(s) from to get the value of this
      side input. If more than one source, then the elements are taken from
      the sources, in the specified order if order matters. At least one
      source is required.
    tag: The id of the tag the user code will access this side input by; this
      should correspond to the tag of some MultiOutputInfo.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class KindValue(_messages.Message):
        """How to interpret the source element(s) as a side input value.

    Messages:
      AdditionalProperty: An additional property for a KindValue object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a KindValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    kind = _messages.MessageField('KindValue', 1)
    sources = _messages.MessageField('Source', 2, repeated=True)
    tag = _messages.StringField(3)