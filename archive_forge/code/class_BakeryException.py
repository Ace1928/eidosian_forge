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
class BakeryException(requests.RequestException):
    """Raised when some errors happen using the httpbakery
    authorizer"""