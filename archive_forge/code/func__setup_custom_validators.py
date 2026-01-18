from __future__ import absolute_import, unicode_literals
import logging
from itertools import chain
from oauthlib.common import add_params_to_uri
from oauthlib.uri_validate import is_absolute_uri
from oauthlib.oauth2.rfc6749 import errors, utils
from ..request_validator import RequestValidator
def _setup_custom_validators(self, kwargs):
    post_auth = kwargs.get('post_auth', [])
    post_token = kwargs.get('post_token', [])
    pre_auth = kwargs.get('pre_auth', [])
    pre_token = kwargs.get('pre_token', [])
    if not hasattr(self, 'validate_authorization_request'):
        if post_auth or pre_auth:
            msg = '{} does not support authorization validators. Use token validators instead.'.format(self.__class__.__name__)
            raise ValueError(msg)
        post_auth, pre_auth = ((), ())
    self.custom_validators = ValidatorsContainer(post_auth, post_token, pre_auth, pre_token)