from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ReportStatsView(_messages.Message):
    """Message to encapsulate the various statistics related to the generated
  Report Next ID: 6

  Fields:
    bytesWritten: Actual size in bytes for the report written, as reported by
      the underlying storage system
    projectNumber: Project Number
    recordsProcessed: Actual records processed as reported by the underlying
      storage system
    reportConfigId: ID of the parent ReportConfig for the corresponding
      ReportDetail
    reportDetailId: ID of the ReportDetail for which the stats are generated
  """
    bytesWritten = _messages.IntegerField(1)
    projectNumber = _messages.IntegerField(2)
    recordsProcessed = _messages.IntegerField(3)
    reportConfigId = _messages.StringField(4)
    reportDetailId = _messages.StringField(5)