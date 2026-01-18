from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ArtifactregistryProjectsLocationsRepositoriesListRequest(_messages.Message):
    """A ArtifactregistryProjectsLocationsRepositoriesListRequest object.

  Fields:
    pageSize: The maximum number of repositories to return. Maximum page size
      is 1,000.
    pageToken: The next_page_token value returned from a previous list
      request, if any.
    parent: Required. The name of the parent resource whose repositories will
      be listed.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)