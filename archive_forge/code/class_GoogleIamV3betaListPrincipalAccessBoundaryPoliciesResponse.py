from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleIamV3betaListPrincipalAccessBoundaryPoliciesResponse(_messages.Message):
    """Response message for ListPrincipalAccessBoundaryPolicies rpc.

  Fields:
    nextPageToken: Optional. A token, which can be sent as `page_token` to
      retrieve the next page. If this field is omitted, there are no
      subsequent pages.
    principalAccessBoundaryPolicies: The principal access boundary policies
      from the specified parent.
  """
    nextPageToken = _messages.StringField(1)
    principalAccessBoundaryPolicies = _messages.MessageField('GoogleIamV3betaPrincipalAccessBoundaryPolicy', 2, repeated=True)