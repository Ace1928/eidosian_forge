from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsApisPatchRequest(_messages.Message):
    """A ApigeeOrganizationsApisPatchRequest object.

  Fields:
    googleCloudApigeeV1ApiProxy: A GoogleCloudApigeeV1ApiProxy resource to be
      passed as the request body.
    name: Required. API proxy to update in the following format:
      `organizations/{org}/apis/{api}`
    updateMask: Required. The list of fields to update.
  """
    googleCloudApigeeV1ApiProxy = _messages.MessageField('GoogleCloudApigeeV1ApiProxy', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)