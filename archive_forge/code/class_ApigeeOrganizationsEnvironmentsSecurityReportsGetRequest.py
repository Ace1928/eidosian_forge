from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsSecurityReportsGetRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsSecurityReportsGetRequest object.

  Fields:
    name: Required. Name of the security report to get. Must be of the form
      `organizations/{org}/environments/{env}/securityReports/{reportId}`.
  """
    name = _messages.StringField(1, required=True)