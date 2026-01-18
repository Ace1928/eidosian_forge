import base64
import json
import logging
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
import macaroonbakery._utils as utils
from ._browser import WebBrowserInteractor
from ._error import (
from ._interactor import (
import requests
from six.moves.http_cookies import SimpleCookie
from six.moves.urllib.parse import urljoin
def _add_json_binary_field(b, serialized, field):
    """' Set the given field to the given val (bytes) in the serialized
    dictionary.
    If the value isn't valid utf-8, we base64 encode it and use field+"64"
    as the field name.
    """
    try:
        val = b.decode('utf-8')
        serialized[field] = val
    except UnicodeDecodeError:
        val = base64.b64encode(b).decode('utf-8')
        serialized[field + '64'] = val