from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsSecurityFeedbackGetRequest(_messages.Message):
    """A ApigeeOrganizationsSecurityFeedbackGetRequest object.

  Fields:
    name: Required. Name of the SecurityFeedback. Format:
      `organizations/{org}/securityFeedback/{feedback_id}` Example:
      organizations/apigee-organization-name/securityFeedback/feedback-id
  """
    name = _messages.StringField(1, required=True)