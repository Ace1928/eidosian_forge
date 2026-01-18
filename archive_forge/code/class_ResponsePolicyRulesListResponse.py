from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ResponsePolicyRulesListResponse(_messages.Message):
    """A ResponsePolicyRulesListResponse object.

  Fields:
    header: A ResponseHeader attribute.
    nextPageToken: The presence of this field indicates that there exist more
      results following your last page of results in pagination order. To
      fetch them, make another list request using this value as your page
      token. This lets you the complete contents of even very large
      collections one page at a time. However, if the contents of the
      collection change between the first and last paginated list request, the
      set of all elements returned are an inconsistent view of the collection.
      You cannot retrieve a consistent snapshot of a collection larger than
      the maximum page size.
    responsePolicyRules: The Response Policy Rule resources.
  """
    header = _messages.MessageField('ResponseHeader', 1)
    nextPageToken = _messages.StringField(2)
    responsePolicyRules = _messages.MessageField('ResponsePolicyRule', 3, repeated=True)