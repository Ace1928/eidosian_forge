from __future__ import absolute_import, division, print_function
from . import errors, http
def _api_key_login(self):
    return dict(Authorization='Key {0}'.format(self.api_key))