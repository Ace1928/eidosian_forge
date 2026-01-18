from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycenterOrganizationsGetSubscriptionRequest(_messages.Message):
    """A SecuritycenterOrganizationsGetSubscriptionRequest object.

  Fields:
    name: Required. The name of the subscription to retrieve. Format:
      organizations/{organization}/subscription
  """
    name = _messages.StringField(1, required=True)