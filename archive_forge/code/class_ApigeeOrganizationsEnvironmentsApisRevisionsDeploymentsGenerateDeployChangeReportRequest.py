from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsApisRevisionsDeploymentsGenerateDeployChangeReportRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsApisRevisionsDeploymentsGenerateDeployC
  hangeReportRequest object.

  Fields:
    name: Name of the API proxy revision deployment in the following format:
      `organizations/{org}/environments/{env}/apis/{api}/revisions/{rev}`
    override: Flag that specifies whether to force the deployment of the new
      revision over the currently deployed revision by overriding conflict
      checks.
  """
    name = _messages.StringField(1, required=True)
    override = _messages.BooleanField(2)