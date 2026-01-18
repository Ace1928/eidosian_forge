from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsGetSecuritySettingsRequest(_messages.Message):
    """A ApigeeOrganizationsGetSecuritySettingsRequest object.

  Fields:
    name: Required. The name of the SecuritySettings to retrieve. This will
      always be: 'organizations/{org}/securitySettings'.
  """
    name = _messages.StringField(1, required=True)