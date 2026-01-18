from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsDevelopersSubscriptionsGetRequest(_messages.Message):
    """A ApigeeOrganizationsDevelopersSubscriptionsGetRequest object.

  Fields:
    name: Required. Name of the API product subscription. Use the following
      structure in your request: `organizations/{org}/developers/{developer_em
      ail}/subscriptions/{subscription}`
  """
    name = _messages.StringField(1, required=True)