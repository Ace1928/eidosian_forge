from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsDeploymentsListRequest(_messages.Message):
    """A ApigeeOrganizationsDeploymentsListRequest object.

  Fields:
    parent: Required. Name of the organization for which to return deployment
      information in the following format: `organizations/{org}`
    sharedFlows: Optional. Flag that specifies whether to return shared flow
      or API proxy deployments. Set to `true` to return shared flow
      deployments; set to `false` to return API proxy deployments. Defaults to
      `false`.
  """
    parent = _messages.StringField(1, required=True)
    sharedFlows = _messages.BooleanField(2)