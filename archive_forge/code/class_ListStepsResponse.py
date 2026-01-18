from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListStepsResponse(_messages.Message):
    """Response message for StepService.List.

  Fields:
    nextPageToken: A continuation token to resume the query at the next item.
      If set, indicates that there are more steps to read, by calling list
      again with this value in the page_token field.
    steps: Steps.
  """
    nextPageToken = _messages.StringField(1)
    steps = _messages.MessageField('Step', 2, repeated=True)