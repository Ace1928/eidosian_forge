from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1WorkerPoolSpec(_messages.Message):
    """Represents the spec of a worker pool in a job.

  Fields:
    containerSpec: The custom container task.
    diskSpec: Disk spec.
    machineSpec: Optional. Immutable. The specification of a single machine.
    nfsMounts: Optional. List of NFS mount spec.
    pythonPackageSpec: The Python packaged task.
    replicaCount: Optional. The number of worker replicas to use for this
      worker pool.
  """
    containerSpec = _messages.MessageField('GoogleCloudAiplatformV1ContainerSpec', 1)
    diskSpec = _messages.MessageField('GoogleCloudAiplatformV1DiskSpec', 2)
    machineSpec = _messages.MessageField('GoogleCloudAiplatformV1MachineSpec', 3)
    nfsMounts = _messages.MessageField('GoogleCloudAiplatformV1NfsMount', 4, repeated=True)
    pythonPackageSpec = _messages.MessageField('GoogleCloudAiplatformV1PythonPackageSpec', 5)
    replicaCount = _messages.IntegerField(6)