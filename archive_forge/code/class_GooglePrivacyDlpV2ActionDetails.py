from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2ActionDetails(_messages.Message):
    """The results of an Action.

  Fields:
    deidentifyDetails: Outcome of a de-identification action.
  """
    deidentifyDetails = _messages.MessageField('GooglePrivacyDlpV2DeidentifyDataSourceDetails', 1)