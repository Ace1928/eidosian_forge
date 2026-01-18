from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class EssentialcontactsOrganizationsContactsResendRequest(_messages.Message):
    """A EssentialcontactsOrganizationsContactsResendRequest object.

  Fields:
    googleCloudEssentialcontactsV1alpha1ResendVerificationRequest: A
      GoogleCloudEssentialcontactsV1alpha1ResendVerificationRequest resource
      to be passed as the request body.
    name: Required. The name of the contact to verify. Format:
      organizations/{organization_id}/contacts/{contact_id},
      folders/{folder_id}/contacts/{contact_id} or
      projects/{project_id}/contacts/{contact_id}
  """
    googleCloudEssentialcontactsV1alpha1ResendVerificationRequest = _messages.MessageField('GoogleCloudEssentialcontactsV1alpha1ResendVerificationRequest', 1)
    name = _messages.StringField(2, required=True)