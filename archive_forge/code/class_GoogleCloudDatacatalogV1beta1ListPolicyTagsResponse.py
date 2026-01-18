from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1beta1ListPolicyTagsResponse(_messages.Message):
    """Response message for ListPolicyTags.

  Fields:
    nextPageToken: Token used to retrieve the next page of results, or empty
      if there are no more results in the list.
    policyTags: The policy tags that are in the requested taxonomy.
  """
    nextPageToken = _messages.StringField(1)
    policyTags = _messages.MessageField('GoogleCloudDatacatalogV1beta1PolicyTag', 2, repeated=True)