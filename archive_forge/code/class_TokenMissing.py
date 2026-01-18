from urllib.parse import urlparse
import logging
from oauthlib.common import add_params_to_uri
from oauthlib.common import urldecode as _urldecode
from oauthlib.oauth1 import SIGNATURE_HMAC, SIGNATURE_RSA, SIGNATURE_TYPE_AUTH_HEADER
import requests
from . import OAuth1
class TokenMissing(ValueError):

    def __init__(self, message, response):
        super(TokenMissing, self).__init__(message)
        self.response = response