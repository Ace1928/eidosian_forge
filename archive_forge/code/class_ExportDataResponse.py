from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ExportDataResponse(_messages.Message):
    """Response for the ExportData method.

  Fields:
    data: The JSON string with customer data and metadata for an execution
      with the given name
  """
    data = _messages.StringField(1)