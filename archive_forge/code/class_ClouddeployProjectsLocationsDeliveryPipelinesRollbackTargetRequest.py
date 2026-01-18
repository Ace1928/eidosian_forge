from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClouddeployProjectsLocationsDeliveryPipelinesRollbackTargetRequest(_messages.Message):
    """A ClouddeployProjectsLocationsDeliveryPipelinesRollbackTargetRequest
  object.

  Fields:
    name: Required. The `DeliveryPipeline` for which the rollback `Rollout`
      should be created. Format should be `projects/{project_id}/locations/{lo
      cation_name}/deliveryPipelines/{pipeline_name}`.
    rollbackTargetRequest: A RollbackTargetRequest resource to be passed as
      the request body.
  """
    name = _messages.StringField(1, required=True)
    rollbackTargetRequest = _messages.MessageField('RollbackTargetRequest', 2)