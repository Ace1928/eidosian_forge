from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ReplicationSpec(_messages.Message):
    """Specifies the configuration for running a replication job.

  Fields:
    gcsDataSink: Specifies cloud Storage data sink.
    gcsDataSource: Specifies cloud Storage data source.
    objectConditions: Specifies the object conditions to only include objects
      that satisfy these conditions in the set of data source objects. Object
      conditions based on objects' "last modification time" do not exclude
      objects in a data sink.
    transferOptions: Specifies the actions to be performed on the object
      during replication. Delete options are not supported for replication and
      when specified, the request fails with an INVALID_ARGUMENT error.
  """
    gcsDataSink = _messages.MessageField('GcsData', 1)
    gcsDataSource = _messages.MessageField('GcsData', 2)
    objectConditions = _messages.MessageField('ObjectConditions', 3)
    transferOptions = _messages.MessageField('TransferOptions', 4)