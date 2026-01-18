from __future__ import absolute_import, unicode_literals
import hashlib
import hmac
from binascii import b2a_base64
import warnings
from oauthlib import common
from oauthlib.common import add_params_to_qs, add_params_to_uri, unicode_type
from . import utils
class OAuth2Token(dict):

    def __init__(self, params, old_scope=None):
        super(OAuth2Token, self).__init__(params)
        self._new_scope = None
        if 'scope' in params and params['scope']:
            self._new_scope = set(utils.scope_to_list(params['scope']))
        if old_scope is not None:
            self._old_scope = set(utils.scope_to_list(old_scope))
            if self._new_scope is None:
                self._new_scope = self._old_scope
        else:
            self._old_scope = self._new_scope

    @property
    def scope_changed(self):
        return self._new_scope != self._old_scope

    @property
    def old_scope(self):
        return utils.list_to_scope(self._old_scope)

    @property
    def old_scopes(self):
        return list(self._old_scope)

    @property
    def scope(self):
        return utils.list_to_scope(self._new_scope)

    @property
    def scopes(self):
        return list(self._new_scope)

    @property
    def missing_scopes(self):
        return list(self._old_scope - self._new_scope)

    @property
    def additional_scopes(self):
        return list(self._new_scope - self._old_scope)