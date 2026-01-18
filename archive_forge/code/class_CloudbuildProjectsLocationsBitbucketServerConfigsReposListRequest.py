from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildProjectsLocationsBitbucketServerConfigsReposListRequest(_messages.Message):
    """A CloudbuildProjectsLocationsBitbucketServerConfigsReposListRequest
  object.

  Fields:
    pageSize: The maximum number of configs to return. The service may return
      fewer than this value. The maximum value is 1000; values above 1000 will
      be coerced to 1000.
    pageToken: A page token, received from a previous
      `ListBitbucketServerRepositoriesRequest` call. Provide this to retrieve
      the subsequent page. When paginating, all other parameters provided to
      `ListBitbucketServerConfigsRequest` must match the call that provided
      the page token.
    parent: Required. Name of the parent resource.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)