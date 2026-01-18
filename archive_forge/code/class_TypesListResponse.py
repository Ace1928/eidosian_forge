from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TypesListResponse(_messages.Message):
    """A response that returns all Types supported by Deployment Manager

  Fields:
    nextPageToken: A token used to continue a truncated list request.
    types: Output only. A list of resource types supported by Deployment
      Manager.
  """
    nextPageToken = _messages.StringField(1)
    types = _messages.MessageField('Type', 2, repeated=True)