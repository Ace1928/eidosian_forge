from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExportDataStatistics(_messages.Message):
    """Statistics for the EXPORT DATA statement as part of Query Job. EXTRACT
  JOB statistics are populated in JobStatistics4.

  Fields:
    fileCount: Number of destination files generated in case of EXPORT DATA
      statement only.
    rowCount: [Alpha] Number of destination rows generated in case of EXPORT
      DATA statement only.
  """
    fileCount = _messages.IntegerField(1)
    rowCount = _messages.IntegerField(2)