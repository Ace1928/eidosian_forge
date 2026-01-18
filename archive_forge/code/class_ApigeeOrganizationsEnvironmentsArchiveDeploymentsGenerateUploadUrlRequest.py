from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsArchiveDeploymentsGenerateUploadUrlRequest(_messages.Message):
    """A
  ApigeeOrganizationsEnvironmentsArchiveDeploymentsGenerateUploadUrlRequest
  object.

  Fields:
    googleCloudApigeeV1GenerateUploadUrlRequest: A
      GoogleCloudApigeeV1GenerateUploadUrlRequest resource to be passed as the
      request body.
    parent: Required. The organization and environment to upload to.
  """
    googleCloudApigeeV1GenerateUploadUrlRequest = _messages.MessageField('GoogleCloudApigeeV1GenerateUploadUrlRequest', 1)
    parent = _messages.StringField(2, required=True)