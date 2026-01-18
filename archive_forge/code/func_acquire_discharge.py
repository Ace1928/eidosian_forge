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
def acquire_discharge(self, cav, payload):
    """ Request a discharge macaroon from the caveat location
        as an HTTP URL.
        @param cav Third party {pymacaroons.Caveat} to be discharged.
        @param payload External caveat data {bytes}.
        @return The acquired macaroon {macaroonbakery.Macaroon}
        """
    resp = self._acquire_discharge_with_token(cav, payload, None)
    if resp.status_code == 200:
        return bakery.Macaroon.from_dict(resp.json().get('Macaroon'))
    try:
        cause = Error.from_dict(resp.json())
    except ValueError:
        raise DischargeError('unexpected response: [{}] {!r}'.format(resp.status_code, resp.content))
    if cause.code != ERR_INTERACTION_REQUIRED:
        raise DischargeError(cause.message)
    if cause.info is None:
        raise DischargeError('interaction-required response with no info: {}'.format(resp.json()))
    loc = cav.location
    if not loc.endswith('/'):
        loc = loc + '/'
    token, m = self._interact(loc, cause, payload)
    if m is not None:
        return m
    resp = self._acquire_discharge_with_token(cav, payload, token)
    if resp.status_code == 200:
        return bakery.Macaroon.from_dict(resp.json().get('Macaroon'))
    else:
        raise DischargeError('discharge failed with code {}'.format(resp.status_code))