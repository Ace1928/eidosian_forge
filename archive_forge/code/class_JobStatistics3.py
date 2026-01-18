from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class JobStatistics3(_messages.Message):
    """A JobStatistics3 object.

  Fields:
    inputFileBytes: [Output-only] Number of bytes of source data in a load
      job.
    inputFiles: [Output-only] Number of source files in a load job.
    outputBytes: [Output-only] Size of the loaded data in bytes. Note that
      while a load job is in the running state, this value may change.
    outputRows: [Output-only] Number of rows imported in a load job. Note that
      while an import job is in the running state, this value may change.
  """
    inputFileBytes = _messages.IntegerField(1)
    inputFiles = _messages.IntegerField(2)
    outputBytes = _messages.IntegerField(3)
    outputRows = _messages.IntegerField(4)