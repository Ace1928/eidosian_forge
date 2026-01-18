from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsDevelopersSetDeveloperStatusRequest(_messages.Message):
    """A ApigeeOrganizationsDevelopersSetDeveloperStatusRequest object.

  Fields:
    action: Status of the developer. Valid values are `active` and `inactive`.
    name: Required. Name of the developer. Use the following structure in your
      request: `organizations/{org}/developers/{developer_id}`
  """
    action = _messages.StringField(1)
    name = _messages.StringField(2, required=True)