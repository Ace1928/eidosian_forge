from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsApisRevisionsDeploymentsGenerateUndeployChangeReportRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsApisRevisionsDeploymentsGenerateUndeplo
  yChangeReportRequest object.

  Fields:
    name: Name of the API proxy revision deployment in the following format:
      `organizations/{org}/environments/{env}/apis/{api}/revisions/{rev}`
  """
    name = _messages.StringField(1, required=True)