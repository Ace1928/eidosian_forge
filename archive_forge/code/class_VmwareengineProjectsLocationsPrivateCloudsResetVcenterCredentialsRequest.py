from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareengineProjectsLocationsPrivateCloudsResetVcenterCredentialsRequest(_messages.Message):
    """A
  VmwareengineProjectsLocationsPrivateCloudsResetVcenterCredentialsRequest
  object.

  Fields:
    privateCloud: Required. The resource name of the private cloud to reset
      credentials for. Resource names are schemeless URIs that follow the
      conventions in https://cloud.google.com/apis/design/resource_names. For
      example: `projects/my-project/locations/us-central1-a/privateClouds/my-
      cloud`
    resetVcenterCredentialsRequest: A ResetVcenterCredentialsRequest resource
      to be passed as the request body.
  """
    privateCloud = _messages.StringField(1, required=True)
    resetVcenterCredentialsRequest = _messages.MessageField('ResetVcenterCredentialsRequest', 2)