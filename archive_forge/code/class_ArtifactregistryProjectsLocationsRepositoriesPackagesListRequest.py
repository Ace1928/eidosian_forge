from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ArtifactregistryProjectsLocationsRepositoriesPackagesListRequest(_messages.Message):
    """A ArtifactregistryProjectsLocationsRepositoriesPackagesListRequest
  object.

  Fields:
    filter: Optional. An expression for filtering the results of the request.
      Filter rules are case insensitive. The fields eligible for filtering
      are: * `name` Examples of using a filter: *
      `name="projects/p1/locations/us-
      central1/repositories/repo1/packages/a%2Fb%2F*"` --> packages with an ID
      starting with "a/b/". * `name="projects/p1/locations/us-
      central1/repositories/repo1/packages/*%2Fb%2Fc"` --> packages with an ID
      ending with "/b/c". * `name="projects/p1/locations/us-
      central1/repositories/repo1/packages/*%2Fb%2F*"` --> packages with an ID
      containing "/b/".
    pageSize: The maximum number of packages to return. Maximum page size is
      1,000.
    pageToken: The next_page_token value returned from a previous list
      request, if any.
    parent: Required. The name of the parent resource whose packages will be
      listed.
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)