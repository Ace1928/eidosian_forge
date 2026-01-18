from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class GoogleCloudOrgpolicyV2ListPoliciesResponse(_messages.Message):
    """The response returned from the ListPolicies method. It will be empty if
  no policies are set on the resource.

  Fields:
    nextPageToken: Page token used to retrieve the next page. This is
      currently not used, but the server may at any point start supplying a
      valid token.
    policies: All policies that exist on the resource. It will be empty if no
      policies are set.
  """
    nextPageToken = _messages.StringField(1)
    policies = _messages.MessageField('GoogleCloudOrgpolicyV2Policy', 2, repeated=True)