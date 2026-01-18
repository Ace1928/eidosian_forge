from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsGetSecurityActionsConfigRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsGetSecurityActionsConfigRequest object.

  Fields:
    name: Required. The name of the SecurityActionsConfig to retrieve. This
      will always be:
      `organizations/{org}/environments/{env}/security_actions_config`
  """
    name = _messages.StringField(1, required=True)