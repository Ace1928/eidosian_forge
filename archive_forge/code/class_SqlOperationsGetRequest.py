from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SqlOperationsGetRequest(_messages.Message):
    """A SqlOperationsGetRequest object.

  Fields:
    operation: Instance operation ID.
    project: Project ID of the project that contains the instance.
  """
    operation = _messages.StringField(1, required=True)
    project = _messages.StringField(2, required=True)