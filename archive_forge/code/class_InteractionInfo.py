import copy
import json
import logging
from collections import namedtuple
import macaroonbakery.bakery as bakery
import macaroonbakery.httpbakery as httpbakery
import macaroonbakery._utils as utils
import requests.cookies
from six.moves.urllib.parse import urljoin
class InteractionInfo(object):
    """Holds the information expected in the agent interaction entry in an
    interaction-required error.
    """

    def __init__(self, login_url):
        self._login_url = login_url

    @property
    def login_url(self):
        """ Return the URL from which to acquire a macaroon that can be used
        to complete the agent login. To acquire the macaroon, make a POST
        request to the URL with user and public-key parameters.
        :return string
        """
        return self._login_url

    @classmethod
    def from_dict(cls, json_dict):
        """Return an InteractionInfo obtained from the given dictionary as
        deserialized from JSON.
        @param json_dict The deserialized JSON object.
        """
        return InteractionInfo(json_dict.get('login-url'))