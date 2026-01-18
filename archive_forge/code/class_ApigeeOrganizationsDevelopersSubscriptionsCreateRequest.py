from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsDevelopersSubscriptionsCreateRequest(_messages.Message):
    """A ApigeeOrganizationsDevelopersSubscriptionsCreateRequest object.

  Fields:
    googleCloudApigeeV1DeveloperSubscription: A
      GoogleCloudApigeeV1DeveloperSubscription resource to be passed as the
      request body.
    parent: Required. Email address of the developer that is purchasing a
      subscription to the API product. Use the following structure in your
      request: `organizations/{org}/developers/{developer_email}`
  """
    googleCloudApigeeV1DeveloperSubscription = _messages.MessageField('GoogleCloudApigeeV1DeveloperSubscription', 1)
    parent = _messages.StringField(2, required=True)