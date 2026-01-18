from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StreamingSetupTask(_messages.Message):
    """A task which initializes part of a streaming Dataflow job.

  Fields:
    drain: The user has requested drain.
    receiveWorkPort: The TCP port on which the worker should listen for
      messages from other streaming computation workers.
    snapshotConfig: Configures streaming appliance snapshot.
    streamingComputationTopology: The global topology of the streaming
      Dataflow job.
    workerHarnessPort: The TCP port used by the worker to communicate with the
      Dataflow worker harness.
  """
    drain = _messages.BooleanField(1)
    receiveWorkPort = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    snapshotConfig = _messages.MessageField('StreamingApplianceSnapshotConfig', 3)
    streamingComputationTopology = _messages.MessageField('TopologyConfig', 4)
    workerHarnessPort = _messages.IntegerField(5, variant=_messages.Variant.INT32)