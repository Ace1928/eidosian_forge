from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StartManualTransferRunsResponse(_messages.Message):
    """A response to start manual transfer runs.

  Fields:
    runs: The transfer runs that were created.
  """
    runs = _messages.MessageField('TransferRun', 1, repeated=True)