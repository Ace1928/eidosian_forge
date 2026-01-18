from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TaskResourceRequest(_messages.Message):
    """Resources used per task created by the application.

  Fields:
    amount: A number attribute.
    resourceName: A string attribute.
  """
    amount = _messages.FloatField(1)
    resourceName = _messages.StringField(2)