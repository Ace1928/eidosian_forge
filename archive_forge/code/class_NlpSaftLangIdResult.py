from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NlpSaftLangIdResult(_messages.Message):
    """A NlpSaftLangIdResult object.

  Enums:
    ModelVersionValueValuesEnum: The version of the model used to create these
      annotations.

  Fields:
    modelVersion: The version of the model used to create these annotations.
    predictions: This field stores the n-best list of possible BCP 47 language
      code strings for a given input sorted in descending order according to
      each code's respective probability.
    spanPredictions: This field stores language predictions of subspans of the
      input, when available. Each LanguageSpanSequence is a sequence of
      LanguageSpans. A particular sequence of LanguageSpans has an associated
      probability, and need not necessarily cover the entire input. If no
      language could be predicted for any span, then this field may be empty.
  """

    class ModelVersionValueValuesEnum(_messages.Enum):
        """The version of the model used to create these annotations.

    Values:
      VERSION_UNSPECIFIED: <no description>
      INDEXING_20181017: <no description>
      INDEXING_20191206: <no description>
      INDEXING_20200313: <no description>
      INDEXING_20210618: <no description>
      STANDARD_20220516: <no description>
    """
        VERSION_UNSPECIFIED = 0
        INDEXING_20181017 = 1
        INDEXING_20191206 = 2
        INDEXING_20200313 = 3
        INDEXING_20210618 = 4
        STANDARD_20220516 = 5
    modelVersion = _messages.EnumField('ModelVersionValueValuesEnum', 1)
    predictions = _messages.MessageField('NlpSaftLanguageSpan', 2, repeated=True)
    spanPredictions = _messages.MessageField('NlpSaftLanguageSpanSequence', 3, repeated=True)