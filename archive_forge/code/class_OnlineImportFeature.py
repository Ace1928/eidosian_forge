from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OnlineImportFeature(_messages.Message):
    """An online import transfer, where data is loaded onto the appliance and
  automatically transferred to Cloud Storage whenever it has an internet
  connection.

  Fields:
    destination: The destination of the transfer.
    jobName: Output only. The Transfer Job Name for Online Imports.
    transferResults: Output only. The results of the transfer.
  """
    destination = _messages.MessageField('GcsDestination', 1)
    jobName = _messages.StringField(2)
    transferResults = _messages.MessageField('TransferResults', 3, repeated=True)