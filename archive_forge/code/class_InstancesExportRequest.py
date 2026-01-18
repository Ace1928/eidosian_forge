from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class InstancesExportRequest(_messages.Message):
    """Database instance export request.

  Fields:
    exportContext: Contains details about the export operation.
  """
    exportContext = _messages.MessageField('ExportContext', 1)