from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListProjectsResponse(_messages.Message):
    """A page of the response received from the ListProjects method. A
  paginated response where more pages are available has `next_page_token` set.
  This token can be used in a subsequent request to retrieve the next request
  page.

  Fields:
    nextPageToken: Pagination token. If the result set is too large to fit in
      a single response, this token is returned. It encodes the position of
      the current result cursor. Feeding this value into a new list request
      with the `page_token` parameter gives the next page of the results. When
      `next_page_token` is not filled in, there is no next page and the list
      returned is the last page in the result set. Pagination tokens have a
      limited lifetime.
    projects: The list of Projects that matched the list filter. This list can
      be paginated.
  """
    nextPageToken = _messages.StringField(1)
    projects = _messages.MessageField('Project', 2, repeated=True)