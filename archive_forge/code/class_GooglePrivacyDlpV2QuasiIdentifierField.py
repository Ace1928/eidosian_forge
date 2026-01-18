from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2QuasiIdentifierField(_messages.Message):
    """A quasi-identifier column has a custom_tag, used to know which column in
  the data corresponds to which column in the statistical model.

  Fields:
    customTag: A column can be tagged with a custom tag. In this case, the
      user must indicate an auxiliary table that contains statistical
      information on the possible values of this column (below).
    field: Identifies the column.
  """
    customTag = _messages.StringField(1)
    field = _messages.MessageField('GooglePrivacyDlpV2FieldId', 2)