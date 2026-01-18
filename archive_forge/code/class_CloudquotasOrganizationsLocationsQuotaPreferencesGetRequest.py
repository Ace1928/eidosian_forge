from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class CloudquotasOrganizationsLocationsQuotaPreferencesGetRequest(_messages.Message):
    """A CloudquotasOrganizationsLocationsQuotaPreferencesGetRequest object.

  Fields:
    name: Required. Name of the resource Example name:
      `projects/123/locations/global/quota_preferences/my-config-for-us-east1`
  """
    name = _messages.StringField(1, required=True)