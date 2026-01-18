from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServicenetworkingServicesDisableVpcServiceControlsRequest(_messages.Message):
    """A ServicenetworkingServicesDisableVpcServiceControlsRequest object.

  Fields:
    disableVpcServiceControlsRequest: A DisableVpcServiceControlsRequest
      resource to be passed as the request body.
    parent: The service that is managing peering connectivity for a service
      producer's organization. For Google services that support this
      functionality, this value is
      `services/servicenetworking.googleapis.com`.
  """
    disableVpcServiceControlsRequest = _messages.MessageField('DisableVpcServiceControlsRequest', 1)
    parent = _messages.StringField(2, required=True)