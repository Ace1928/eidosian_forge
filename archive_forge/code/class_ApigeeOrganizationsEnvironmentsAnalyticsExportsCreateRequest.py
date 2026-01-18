from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsAnalyticsExportsCreateRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsAnalyticsExportsCreateRequest object.

  Fields:
    googleCloudApigeeV1ExportRequest: A GoogleCloudApigeeV1ExportRequest
      resource to be passed as the request body.
    parent: Required. Names of the parent organization and environment. Must
      be of the form `organizations/{org}/environments/{env}`.
  """
    googleCloudApigeeV1ExportRequest = _messages.MessageField('GoogleCloudApigeeV1ExportRequest', 1)
    parent = _messages.StringField(2, required=True)