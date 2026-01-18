from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsSecurityReportsCreateRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsSecurityReportsCreateRequest object.

  Fields:
    googleCloudApigeeV1SecurityReportQuery: A
      GoogleCloudApigeeV1SecurityReportQuery resource to be passed as the
      request body.
    parent: Required. The parent resource name. Must be of the form
      `organizations/{org}/environments/{env}`.
  """
    googleCloudApigeeV1SecurityReportQuery = _messages.MessageField('GoogleCloudApigeeV1SecurityReportQuery', 1)
    parent = _messages.StringField(2, required=True)