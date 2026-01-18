from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsGetRuntimeConfigRequest(_messages.Message):
    """A ApigeeOrganizationsGetRuntimeConfigRequest object.

  Fields:
    name: Required. Name of the runtime config for the organization in the
      following format: 'organizations/{org}/runtimeConfig'.
  """
    name = _messages.StringField(1, required=True)