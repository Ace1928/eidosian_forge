from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsGetApiSecurityRuntimeConfigRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsGetApiSecurityRuntimeConfigRequest
  object.

  Fields:
    name: Required. Name of the environment API Security Runtime configuration
      resource. Use the following structure in your request:
      `organizations/{org}/environments/{env}/apiSecurityRuntimeConfig`
  """
    name = _messages.StringField(1, required=True)