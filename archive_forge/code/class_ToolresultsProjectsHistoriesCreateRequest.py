from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ToolresultsProjectsHistoriesCreateRequest(_messages.Message):
    """A ToolresultsProjectsHistoriesCreateRequest object.

  Fields:
    history: A History resource to be passed as the request body.
    projectId: A Project id. Required.
    requestId: A unique request ID for server to detect duplicated requests.
      For example, a UUID. Optional, but strongly recommended.
  """
    history = _messages.MessageField('History', 1)
    projectId = _messages.StringField(2, required=True)
    requestId = _messages.StringField(3)