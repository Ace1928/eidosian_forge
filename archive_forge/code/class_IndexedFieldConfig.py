from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IndexedFieldConfig(_messages.Message):
    """A IndexedFieldConfig object.

  Enums:
    StateValueValuesEnum: Output only. State shows whether IndexedFieldConfig
      is ready to be used.
    TokenizationValueValuesEnum: String tokenization mode.

  Fields:
    expression: Expression to evaluate in the context of an indexed property
      defaults to the property name.
    fullTextSearch: Enable full text search - strings only. DEPRECATED, use
      tokenization
    state: Output only. State shows whether IndexedFieldConfig is ready to be
      used.
    tokenization: String tokenization mode.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. State shows whether IndexedFieldConfig is ready to be
    used.

    Values:
      STATE_UNSPECIFIED: STATE_UNSPECIFIED is not an expected state.
      UPDATING: UPDATING means the config is still being updated and not
        filterable.
      ACTIVE: ACTIVE means the config is filterable.
    """
        STATE_UNSPECIFIED = 0
        UPDATING = 1
        ACTIVE = 2

    class TokenizationValueValuesEnum(_messages.Enum):
        """String tokenization mode.

    Values:
      TOKENIZATION_UNSPECIFIED: No tokenization - only exact string matches
        are supported.
      WORDS: Use word tokens.
      SUBSTRINGS_NGRAM_3: Uses 3-ngram tokens supporting efficient substring
        searches.
    """
        TOKENIZATION_UNSPECIFIED = 0
        WORDS = 1
        SUBSTRINGS_NGRAM_3 = 2
    expression = _messages.StringField(1)
    fullTextSearch = _messages.BooleanField(2)
    state = _messages.EnumField('StateValueValuesEnum', 3)
    tokenization = _messages.EnumField('TokenizationValueValuesEnum', 4)