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
def _get_external_param_name(self, param_name):
    """Named operation parameter names are sent as is."""
    return param_name