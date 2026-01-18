from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1MutateDeployedModelRequest(_messages.Message):
    """Request message for EndpointService.MutateDeployedModel.

  Fields:
    deployedModel: Required. The DeployedModel to be mutated within the
      Endpoint. Only the following fields can be mutated: *
      `min_replica_count` in either DedicatedResources or AutomaticResources *
      `max_replica_count` in either DedicatedResources or AutomaticResources *
      autoscaling_metric_specs * `disable_container_logging` (v1 only) *
      `enable_container_logging` (v1beta1 only)
    updateMask: Required. The update mask applies to the resource. See
      google.protobuf.FieldMask.
  """
    deployedModel = _messages.MessageField('GoogleCloudAiplatformV1beta1DeployedModel', 1)
    updateMask = _messages.StringField(2)