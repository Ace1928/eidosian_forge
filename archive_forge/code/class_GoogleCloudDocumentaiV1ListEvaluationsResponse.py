from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1ListEvaluationsResponse(_messages.Message):
    """The response from `ListEvaluations`.

  Fields:
    evaluations: The evaluations requested.
    nextPageToken: A token, which can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
  """
    evaluations = _messages.MessageField('GoogleCloudDocumentaiV1Evaluation', 1, repeated=True)
    nextPageToken = _messages.StringField(2)