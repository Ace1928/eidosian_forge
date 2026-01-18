from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListPostureTemplatesResponse(_messages.Message):
    """Message for response to listing PostureTemplates.

  Fields:
    nextPageToken: A token identifying a page of results the server should
      return.
    postureTemplates: The list of PostureTemplate.
  """
    nextPageToken = _messages.StringField(1)
    postureTemplates = _messages.MessageField('PostureTemplate', 2, repeated=True)