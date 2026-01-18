from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2ListDeidentifyTemplatesResponse(_messages.Message):
    """Response message for ListDeidentifyTemplates.

  Fields:
    deidentifyTemplates: List of deidentify templates, up to page_size in
      ListDeidentifyTemplatesRequest.
    nextPageToken: If the next page is available then the next page token to
      be used in the following ListDeidentifyTemplates request.
  """
    deidentifyTemplates = _messages.MessageField('GooglePrivacyDlpV2DeidentifyTemplate', 1, repeated=True)
    nextPageToken = _messages.StringField(2)