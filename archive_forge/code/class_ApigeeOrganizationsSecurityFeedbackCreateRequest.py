from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsSecurityFeedbackCreateRequest(_messages.Message):
    """A ApigeeOrganizationsSecurityFeedbackCreateRequest object.

  Fields:
    googleCloudApigeeV1SecurityFeedback: A GoogleCloudApigeeV1SecurityFeedback
      resource to be passed as the request body.
    parent: Required. Name of the organization. Use the following structure in
      your request: `organizations/{org}`.
    securityFeedbackId: Optional. The id for this feedback report. If not
      provided, it will be set to a system-generated UUID.
  """
    googleCloudApigeeV1SecurityFeedback = _messages.MessageField('GoogleCloudApigeeV1SecurityFeedback', 1)
    parent = _messages.StringField(2, required=True)
    securityFeedbackId = _messages.StringField(3)