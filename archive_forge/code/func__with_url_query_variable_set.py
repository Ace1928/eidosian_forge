from email.message import Message
from io import BytesIO
from json import dumps, loads
import sys
from wadllib.application import Resource as WadlResource
from lazr.restfulclient import __version__
from lazr.restfulclient._browser import Browser, RestfulHttp
from lazr.restfulclient._json import DatetimeJSONEncoder
from lazr.restfulclient.errors import HTTPError
from lazr.uri import URI
def _with_url_query_variable_set(self, url, variable, new_value):
    """A helper method to set a query variable in a URL."""
    uri = URI(url)
    if uri.query is None:
        params = {}
    else:
        params = parse_qs(uri.query)
    params[variable] = str(new_value)
    uri.query = urlencode(params, True)
    return str(uri)