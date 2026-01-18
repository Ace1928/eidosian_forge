from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsSharedflowsRevisionsUndeployRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsSharedflowsRevisionsUndeployRequest
  object.

  Fields:
    name: Required. Name of the shared flow revision to undeploy in the
      following format: `organizations/{org}/environments/{env}/sharedflows/{s
      haredflow}/revisions/{rev}`
  """
    name = _messages.StringField(1, required=True)