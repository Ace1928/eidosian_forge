from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecretmanagerProjectsSecretsListRequest(_messages.Message):
    """A SecretmanagerProjectsSecretsListRequest object.

  Fields:
    filter: Optional. Filter string, adhering to the rules in [List-operation
      filtering](https://cloud.google.com/secret-manager/docs/filtering). List
      only secrets matching the filter. If filter is empty, all secrets are
      listed.
    pageSize: Optional. The maximum number of results to be returned in a
      single page. If set to 0, the server decides the number of results to
      return. If the number is greater than 25000, it is capped at 25000.
    pageToken: Optional. Pagination token, returned earlier via
      ListSecretsResponse.next_page_token.
    parent: Required. The resource name of the project associated with the
      Secrets, in the format `projects/*` or `projects/*/locations/*`
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)