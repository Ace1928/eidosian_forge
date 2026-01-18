from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SyncError(_messages.Message):
    """An ACM created error representing a problem syncing configurations

  Fields:
    code: An ACM defined error code
    errorMessage: A description of the error
    errorResources: A list of config(s) associated with the error, if any
  """
    code = _messages.StringField(1)
    errorMessage = _messages.StringField(2)
    errorResources = _messages.MessageField('ErrorResource', 3, repeated=True)