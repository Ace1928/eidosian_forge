from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2QuasiId(_messages.Message):
    """A column with a semantic tag attached.

  Fields:
    customTag: A column can be tagged with a custom tag. In this case, the
      user must indicate an auxiliary table that contains statistical
      information on the possible values of this column (below).
    field: Required. Identifies the column.
    inferred: If no semantic tag is indicated, we infer the statistical model
      from the distribution of values in the input data
    infoType: A column can be tagged with a InfoType to use the relevant
      public dataset as a statistical model of population, if available. We
      currently support US ZIP codes, region codes, ages and genders. To
      programmatically obtain the list of supported InfoTypes, use
      ListInfoTypes with the supported_by=RISK_ANALYSIS filter.
  """
    customTag = _messages.StringField(1)
    field = _messages.MessageField('GooglePrivacyDlpV2FieldId', 2)
    inferred = _messages.MessageField('GoogleProtobufEmpty', 3)
    infoType = _messages.MessageField('GooglePrivacyDlpV2InfoType', 4)