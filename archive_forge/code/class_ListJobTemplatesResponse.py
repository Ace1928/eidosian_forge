from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListJobTemplatesResponse(_messages.Message):
    """Response message for `TranscoderService.ListJobTemplates`.

  Fields:
    jobTemplates: List of job templates in the specified region.
    nextPageToken: The pagination token.
    unreachable: List of regions that could not be reached.
  """
    jobTemplates = _messages.MessageField('JobTemplate', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    unreachable = _messages.StringField(3, repeated=True)