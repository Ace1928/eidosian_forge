from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsFeatureOnlineStoresFeatureViewsTestIamPermissionsRequest(_messages.Message):
    """A AiplatformProjectsLocationsFeatureOnlineStoresFeatureViewsTestIamPermi
  ssionsRequest object.

  Fields:
    permissions: The set of permissions to check for the `resource`.
      Permissions with wildcards (such as `*` or `storage.*`) are not allowed.
      For more information see [IAM
      Overview](https://cloud.google.com/iam/docs/overview#permissions).
    resource: REQUIRED: The resource for which the policy detail is being
      requested. See [Resource
      names](https://cloud.google.com/apis/design/resource_names) for the
      appropriate value for this field.
  """
    permissions = _messages.StringField(1, repeated=True)
    resource = _messages.StringField(2, required=True)