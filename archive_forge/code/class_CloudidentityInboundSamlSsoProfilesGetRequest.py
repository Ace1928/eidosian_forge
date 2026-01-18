from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudidentityInboundSamlSsoProfilesGetRequest(_messages.Message):
    """A CloudidentityInboundSamlSsoProfilesGetRequest object.

  Fields:
    name: Required. The [resource
      name](https://cloud.google.com/apis/design/resource_names) of the
      InboundSamlSsoProfile to get. Format:
      `inboundSamlSsoProfiles/{sso_profile_id}`
  """
    name = _messages.StringField(1, required=True)