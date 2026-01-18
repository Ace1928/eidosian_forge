from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AnalyzeEntitiesRequest(_messages.Message):
    """The entity analysis request message.

  Enums:
    EncodingTypeValueValuesEnum: The encoding type used by the API to
      calculate offsets.

  Fields:
    document: Required. Input document.
    encodingType: The encoding type used by the API to calculate offsets.
  """

    class EncodingTypeValueValuesEnum(_messages.Enum):
        """The encoding type used by the API to calculate offsets.

    Values:
      NONE: If `EncodingType` is not specified, encoding-dependent information
        (such as `begin_offset`) will be set at `-1`.
      UTF8: Encoding-dependent information (such as `begin_offset`) is
        calculated based on the UTF-8 encoding of the input. C++ and Go are
        examples of languages that use this encoding natively.
      UTF16: Encoding-dependent information (such as `begin_offset`) is
        calculated based on the UTF-16 encoding of the input. Java and
        JavaScript are examples of languages that use this encoding natively.
      UTF32: Encoding-dependent information (such as `begin_offset`) is
        calculated based on the UTF-32 encoding of the input. Python is an
        example of a language that uses this encoding natively.
    """
        NONE = 0
        UTF8 = 1
        UTF16 = 2
        UTF32 = 3
    document = _messages.MessageField('Document', 1)
    encodingType = _messages.EnumField('EncodingTypeValueValuesEnum', 2)