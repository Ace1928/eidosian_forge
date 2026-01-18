from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsSecurityReportsGetResultViewRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsSecurityReportsGetResultViewRequest
  object.

  Fields:
    name: Required. Name of the security report result view to get. Must be of
      the form `organizations/{org}/environments/{env}/securityReports/{report
      Id}/resultView`.
  """
    name = _messages.StringField(1, required=True)