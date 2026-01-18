from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DiagnoseClusterResults(_messages.Message):
    """The location of diagnostic output.

  Fields:
    outputUri: Output only. The Cloud Storage URI of the diagnostic output.
      The output report is a plain text file with a summary of collected
      diagnostics.
  """
    outputUri = _messages.StringField(1)