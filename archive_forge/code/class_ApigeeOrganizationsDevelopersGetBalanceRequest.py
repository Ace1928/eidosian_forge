from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsDevelopersGetBalanceRequest(_messages.Message):
    """A ApigeeOrganizationsDevelopersGetBalanceRequest object.

  Fields:
    name: Required. Account balance for the developer. Use the following
      structure in your request:
      `organizations/{org}/developers/{developer}/balance`
  """
    name = _messages.StringField(1, required=True)