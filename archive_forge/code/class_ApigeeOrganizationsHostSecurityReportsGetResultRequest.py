from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsHostSecurityReportsGetResultRequest(_messages.Message):
    """A ApigeeOrganizationsHostSecurityReportsGetResultRequest object.

  Fields:
    name: Required. Name of the security report result to get. Must be of the
      form `organizations/{org}/securityReports/{reportId}/result`.
  """
    name = _messages.StringField(1, required=True)