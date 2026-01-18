from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListSessionTemplatesResponse(_messages.Message):
    """A list of session templates.

  Fields:
    nextPageToken: A token, which can be sent as page_token to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
    sessionTemplates: Output only. Session template list
  """
    nextPageToken = _messages.StringField(1)
    sessionTemplates = _messages.MessageField('SessionTemplate', 2, repeated=True)