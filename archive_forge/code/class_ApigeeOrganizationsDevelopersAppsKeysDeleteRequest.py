from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsDevelopersAppsKeysDeleteRequest(_messages.Message):
    """A ApigeeOrganizationsDevelopersAppsKeysDeleteRequest object.

  Fields:
    name: Name of the developer app key. Use the following structure in your
      request:
      `organizations/{org}/developers/{developer_email}/apps/{app}/keys/{key}`
  """
    name = _messages.StringField(1, required=True)