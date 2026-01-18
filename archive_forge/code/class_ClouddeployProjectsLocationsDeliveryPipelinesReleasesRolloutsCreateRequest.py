from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClouddeployProjectsLocationsDeliveryPipelinesReleasesRolloutsCreateRequest(_messages.Message):
    """A
  ClouddeployProjectsLocationsDeliveryPipelinesReleasesRolloutsCreateRequest
  object.

  Fields:
    overrideDeployPolicy: Optional. Deploy policies to override. Format is
      `projects/{project}/locations/{location}/deployPolicies/a-z{0,62}`.
    parent: Required. The parent collection in which the `Rollout` should be
      created. Format should be `projects/{project_id}/locations/{location_nam
      e}/deliveryPipelines/{pipeline_name}/releases/{release_name}`.
    requestId: Optional. A request ID to identify requests. Specify a unique
      request ID so that if you must retry your request, the server knows to
      ignore the request if it has already been completed. The server
      guarantees that for at least 60 minutes after the first request. For
      example, consider a situation where you make an initial request and the
      request times out. If you make the request again with the same request
      ID, the server can check if original operation with the same request ID
      was received, and if so, will ignore the second request. This prevents
      clients from accidentally creating duplicate commitments. The request ID
      must be a valid UUID with the exception that zero UUID is not supported
      (00000000-0000-0000-0000-000000000000).
    rollout: A Rollout resource to be passed as the request body.
    rolloutId: Required. ID of the `Rollout`.
    startingPhaseId: Optional. The starting phase ID for the `Rollout`. If
      empty the `Rollout` will start at the first phase.
    validateOnly: Optional. If set to true, the request is validated and the
      user is provided with an expected result, but no actual change is made.
  """
    overrideDeployPolicy = _messages.StringField(1, repeated=True)
    parent = _messages.StringField(2, required=True)
    requestId = _messages.StringField(3)
    rollout = _messages.MessageField('Rollout', 4)
    rolloutId = _messages.StringField(5)
    startingPhaseId = _messages.StringField(6)
    validateOnly = _messages.BooleanField(7)