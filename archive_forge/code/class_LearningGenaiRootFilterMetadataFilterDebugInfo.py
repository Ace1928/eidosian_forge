from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LearningGenaiRootFilterMetadataFilterDebugInfo(_messages.Message):
    """A LearningGenaiRootFilterMetadataFilterDebugInfo object.

  Fields:
    classifierOutput: A LearningGenaiRootClassifierOutput attribute.
    defaultMetadata: A string attribute.
    languageFilterResult: A LearningGenaiRootLanguageFilterResult attribute.
    raiOutput: Safety filter output information for LLM Root RAI harm check.
    raiResult: A CloudAiNlLlmProtoServiceRaiResult attribute.
    raiSignal: A CloudAiNlLlmProtoServiceRaiSignal attribute.
    records: Number of rewinds by controlled decoding.
    streamRecitationResult: A
      LanguageLabsAidaTrustRecitationProtoStreamRecitationResult attribute.
    takedownResult: A LearningGenaiRootTakedownResult attribute.
    toxicityResult: A LearningGenaiRootToxicityResult attribute.
  """
    classifierOutput = _messages.MessageField('LearningGenaiRootClassifierOutput', 1)
    defaultMetadata = _messages.StringField(2)
    languageFilterResult = _messages.MessageField('LearningGenaiRootLanguageFilterResult', 3)
    raiOutput = _messages.MessageField('LearningGenaiRootRAIOutput', 4)
    raiResult = _messages.MessageField('CloudAiNlLlmProtoServiceRaiResult', 5)
    raiSignal = _messages.MessageField('CloudAiNlLlmProtoServiceRaiSignal', 6)
    records = _messages.MessageField('LearningGenaiRootControlDecodingRecords', 7)
    streamRecitationResult = _messages.MessageField('LanguageLabsAidaTrustRecitationProtoStreamRecitationResult', 8)
    takedownResult = _messages.MessageField('LearningGenaiRootTakedownResult', 9)
    toxicityResult = _messages.MessageField('LearningGenaiRootToxicityResult', 10)