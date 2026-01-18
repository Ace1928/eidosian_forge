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
def _wait_for_macaroon(wait_url):
    """ Returns a macaroon from a legacy wait endpoint.
    """
    headers = {BAKERY_PROTOCOL_HEADER: str(bakery.LATEST_VERSION)}
    resp = requests.get(url=wait_url, headers=headers)
    if resp.status_code != 200:
        raise InteractionError('cannot get {}'.format(wait_url))
    return bakery.Macaroon.from_dict(resp.json().get('Macaroon'))