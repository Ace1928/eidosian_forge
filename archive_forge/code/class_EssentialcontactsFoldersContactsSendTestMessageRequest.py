from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class EssentialcontactsFoldersContactsSendTestMessageRequest(_messages.Message):
    """A EssentialcontactsFoldersContactsSendTestMessageRequest object.

  Fields:
    googleCloudEssentialcontactsV1SendTestMessageRequest: A
      GoogleCloudEssentialcontactsV1SendTestMessageRequest resource to be
      passed as the request body.
    resource: Required. The name of the resource to send the test message for.
      All contacts must either be set directly on this resource or inherited
      from another resource that is an ancestor of this one. Format:
      organizations/{organization_id}, folders/{folder_id} or
      projects/{project_id}
  """
    googleCloudEssentialcontactsV1SendTestMessageRequest = _messages.MessageField('GoogleCloudEssentialcontactsV1SendTestMessageRequest', 1)
    resource = _messages.StringField(2, required=True)