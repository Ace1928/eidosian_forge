from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MediaassetProjectsLocationsAssetTypesTestIamPermissionsRequest(_messages.Message):
    """A MediaassetProjectsLocationsAssetTypesTestIamPermissionsRequest object.

  Fields:
    googleIamV1TestIamPermissionsRequest: A
      GoogleIamV1TestIamPermissionsRequest resource to be passed as the
      request body.
    resource: REQUIRED: The resource for which the policy detail is being
      requested. See [Resource
      names](https://cloud.google.com/apis/design/resource_names) for the
      appropriate value for this field.
  """
    googleIamV1TestIamPermissionsRequest = _messages.MessageField('GoogleIamV1TestIamPermissionsRequest', 1)
    resource = _messages.StringField(2, required=True)