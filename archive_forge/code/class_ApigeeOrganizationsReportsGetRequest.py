from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsReportsGetRequest(_messages.Message):
    """A ApigeeOrganizationsReportsGetRequest object.

  Fields:
    name: Required. Custom Report name of the form:
      `organizations/{organization_id}/reports/{report_name}`
  """
    name = _messages.StringField(1, required=True)