import copy
from urllib import parse as urlparse
from magnumclient.common.apiclient import base
def _list_pagination(self, url, response_key=None, obj_class=None, limit=None):
    """Retrieve a list of items.

        The Magnum API is configured to return a maximum number of
        items per request, (FIXME: see Magnum's api.max_limit option). This
        iterates over the 'next' link (pagination) in the responses,
        to get the number of items specified by 'limit'. If 'limit'
        is None this function will continue pagination until there are
        no more values to be returned.

        :param url: a partial URL, e.g. '/nodes'
        :param response_key: the key to be looked up in response
            dictionary, e.g. 'nodes'
        :param obj_class: class for constructing the returned objects.
        :param limit: maximum number of items to return. If None returns
            everything.

        """
    if obj_class is None:
        obj_class = self.resource_class
    if limit is not None:
        limit = int(limit)
    object_list = []
    object_count = 0
    limit_reached = False
    while url:
        resp, body = self.api.json_request('GET', url)
        data = self._format_body_data(body, response_key)
        for obj in data:
            object_list.append(obj_class(self, obj, loaded=True))
            object_count += 1
            if limit and object_count >= limit:
                limit_reached = True
                break
        if limit_reached:
            break
        url = body.get('next')
        if url:
            url_parts = list(urlparse.urlparse(url))
            url_parts[0] = url_parts[1] = ''
            url = urlparse.urlunparse(url_parts)
    return object_list