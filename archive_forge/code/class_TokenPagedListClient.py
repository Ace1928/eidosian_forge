from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.apigee import request
class TokenPagedListClient(PagedListClient):
    """Client for paged `List` APIs that identify pages using a page token.

  This is the AIP-approved way to paginate results and is preferred for new
  APIs.

  Attributes:
    _page_token_field: the field name in the List API's response that contains
      an explicit page token.
    _list_container: the field name in the List API's response that contains the
      list of objects.
    _page_token_param: the query parameter for the previous page's token.
    _max_per_page: the maximum number of items that can be returned in each List
      response.
    _limit_param: the query parameter for the number of items to be returned on
      each page.
  """
    _page_token_field = 'nextPageToken'
    _page_token_param = 'pageToken'
    _max_per_page = 100
    _limit_param = 'pageSize'

    @classmethod
    def List(cls, identifiers=None, extra_params=None):
        if cls._list_container is None:
            error = '%s does not specify a _list_container, but token pagination requires it' % cls
            raise AssertionError(error)
        params = {cls._limit_param: cls._max_per_page}
        if extra_params:
            params.update(extra_params)
        while True:
            response = super(TokenPagedListClient, cls).List(identifiers, params)
            for item in cls._NormalizedResultChunk(response):
                yield item
            if cls._page_token_field in response and response[cls._page_token_field]:
                params[cls._page_token_param] = response[cls._page_token_field]
                continue
            break