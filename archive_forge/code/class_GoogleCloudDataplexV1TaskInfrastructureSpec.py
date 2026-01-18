from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1TaskInfrastructureSpec(_messages.Message):
    """Configuration for the underlying infrastructure used to run workloads.

  Fields:
    batch: Compute resources needed for a Task when using Dataproc Serverless.
    containerImage: Container Image Runtime Configuration.
    vpcNetwork: Vpc network.
  """
    batch = _messages.MessageField('GoogleCloudDataplexV1TaskInfrastructureSpecBatchComputeResources', 1)
    containerImage = _messages.MessageField('GoogleCloudDataplexV1TaskInfrastructureSpecContainerImageRuntime', 2)
    vpcNetwork = _messages.MessageField('GoogleCloudDataplexV1TaskInfrastructureSpecVpcNetwork', 3)