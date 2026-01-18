from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsDevelopersGetRequest(_messages.Message):
    """A ApigeeOrganizationsDevelopersGetRequest object.

  Fields:
    action: Status of the developer. Valid values are `active` or `inactive`.
    name: Required. Email address of the developer. Use the following
      structure in your request:
      `organizations/{org}/developers/{developer_email}`
  """
    action = _messages.StringField(1)
    name = _messages.StringField(2, required=True)