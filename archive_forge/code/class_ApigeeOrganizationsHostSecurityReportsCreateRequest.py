from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsHostSecurityReportsCreateRequest(_messages.Message):
    """A ApigeeOrganizationsHostSecurityReportsCreateRequest object.

  Fields:
    googleCloudApigeeV1SecurityReportQuery: A
      GoogleCloudApigeeV1SecurityReportQuery resource to be passed as the
      request body.
    parent: Required. The parent resource name. Must be of the form
      `organizations/{org}`.
  """
    googleCloudApigeeV1SecurityReportQuery = _messages.MessageField('GoogleCloudApigeeV1SecurityReportQuery', 1)
    parent = _messages.StringField(2, required=True)