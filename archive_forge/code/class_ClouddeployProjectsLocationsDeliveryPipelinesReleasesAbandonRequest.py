from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClouddeployProjectsLocationsDeliveryPipelinesReleasesAbandonRequest(_messages.Message):
    """A ClouddeployProjectsLocationsDeliveryPipelinesReleasesAbandonRequest
  object.

  Fields:
    abandonReleaseRequest: A AbandonReleaseRequest resource to be passed as
      the request body.
    name: Required. Name of the Release. Format is `projects/{project}/locatio
      ns/{location}/deliveryPipelines/{deliveryPipeline}/releases/{release}`.
  """
    abandonReleaseRequest = _messages.MessageField('AbandonReleaseRequest', 1)
    name = _messages.StringField(2, required=True)