from __future__ import absolute_import, unicode_literals
import hashlib
import hmac
from binascii import b2a_base64
import warnings
from oauthlib import common
from oauthlib.common import add_params_to_qs, add_params_to_uri, unicode_type
from . import utils
class TokenBase(object):

    def __call__(self, request, refresh_token=False):
        raise NotImplementedError('Subclasses must implement this method.')

    def validate_request(self, request):
        """
        :param request: OAuthlib request.
        :type request: oauthlib.common.Request
        """
        raise NotImplementedError('Subclasses must implement this method.')

    def estimate_type(self, request):
        """
        :param request: OAuthlib request.
        :type request: oauthlib.common.Request
        """
        raise NotImplementedError('Subclasses must implement this method.')