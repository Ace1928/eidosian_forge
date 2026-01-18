from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.apigee import request
class FieldPagedListClient(PagedListClient):
    """Client for paged `List` APIs that identify pages using a page field.

  This is the pagination method used by legacy Apigee CG APIs, and has been
  preserved for backwards compatibility in Apigee's GCP offering.

  Attributes:
    _list_container: the field name in the List API's response that contains the
      list of objects. None if the API returns a list directly.
    _page_field: the field name in each list element that can be used as a page
      identifier. PageListClient will take the value of this field in the last
      list item for a page, and use it as the  _start_at_param for the next
      page. None if each list element is a primitive which can be used for this
      purpose directly.
    _max_per_page: the maximum number of items that can be returned in each List
      response.
    _limit_param: the query parameter for the number of items to be returned on
      each page.
    _start_at_param: the query parameter for where in the available data the
      response should begin.
  """
    _page_field = None
    _max_per_page = 1000
    _limit_param = 'count'
    _start_at_param = 'startKey'

    @classmethod
    def List(cls, identifiers=None, start_at_param=None, extra_params=None):
        if start_at_param is None:
            start_at_param = cls._start_at_param
        params = {cls._limit_param: cls._max_per_page}
        if extra_params:
            params.update(extra_params)
        while True:
            result_chunk = super(FieldPagedListClient, cls).List(identifiers, params)
            if not result_chunk and start_at_param not in params:
                return
            if cls._list_container is not None:
                result_chunk = cls._NormalizedResultChunk(result_chunk)
            for item in result_chunk[:cls._max_per_page - 1]:
                yield item
            if len(result_chunk) < cls._max_per_page:
                break
            last_item_on_page = result_chunk[-1]
            if cls._page_field is not None:
                last_item_on_page = last_item_on_page[cls._page_field]
            params[start_at_param] = last_item_on_page