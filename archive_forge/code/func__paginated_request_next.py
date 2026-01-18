import base64
import warnings
from libcloud.pricing import get_size_price
from libcloud.utils.py3 import ET, b, next, httplib, parse_qs, urlparse
from libcloud.utils.xml import findall
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.iso8601 import parse_date
from libcloud.common.openstack import (
from libcloud.utils.networking import is_public_subnet
from libcloud.common.exceptions import BaseHTTPError
def _paginated_request_next(self, path, request_method, response_key):
    """
        Perform multiple calls and retrieve all the elements for a paginated
        response.

        This method utilizes "next" attribute in the response object.

        It also includes an infinite loop protection (if the "next" value
        matches the current path, it will abort).

        :param request_method: Method to call which will send the request and
                               return a response. This method will get passed
                               in "path" as a first argument.

        :param response_key: Key in the response object dictionary which
                             contains actual objects we are interested in.
        """
    iteration_count = 0
    result = []
    while path:
        response = request_method(path)
        items = response.object.get(response_key, []) or []
        result.extend(items)
        next_path = response.object.get('next', None)
        if next_path == path:
            break
        if iteration_count > PAGINATION_LIMIT:
            raise OpenStackException('Pagination limit reached for %s, the limit is %d. This might indicate that your API is returning a looping next target for pagination!' % (path, PAGINATION_LIMIT), None)
        path = next_path
        iteration_count += 1
    return result