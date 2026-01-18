from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsDevelopersUpdateMonetizationConfigRequest(_messages.Message):
    """A ApigeeOrganizationsDevelopersUpdateMonetizationConfigRequest object.

  Fields:
    googleCloudApigeeV1DeveloperMonetizationConfig: A
      GoogleCloudApigeeV1DeveloperMonetizationConfig resource to be passed as
      the request body.
    name: Required. Monetization configuration for the developer. Use the
      following structure in your request:
      `organizations/{org}/developers/{developer}/monetizationConfig`
  """
    googleCloudApigeeV1DeveloperMonetizationConfig = _messages.MessageField('GoogleCloudApigeeV1DeveloperMonetizationConfig', 1)
    name = _messages.StringField(2, required=True)