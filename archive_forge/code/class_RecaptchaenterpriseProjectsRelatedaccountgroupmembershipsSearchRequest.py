from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RecaptchaenterpriseProjectsRelatedaccountgroupmembershipsSearchRequest(_messages.Message):
    """A RecaptchaenterpriseProjectsRelatedaccountgroupmembershipsSearchRequest
  object.

  Fields:
    googleCloudRecaptchaenterpriseV1SearchRelatedAccountGroupMembershipsReques
      t: A GoogleCloudRecaptchaenterpriseV1SearchRelatedAccountGroupMembership
      sRequest resource to be passed as the request body.
    project: Required. The name of the project to search related account group
      memberships from. Specify the project name in the following format:
      `projects/{project}`.
  """
    googleCloudRecaptchaenterpriseV1SearchRelatedAccountGroupMembershipsRequest = _messages.MessageField('GoogleCloudRecaptchaenterpriseV1SearchRelatedAccountGroupMembershipsRequest', 1)
    project = _messages.StringField(2, required=True)