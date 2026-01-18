from keystone.common import provider_api
from keystone import exception
from keystone.oauth1.backends import base
from keystone.oauth1 import core as oauth1
def _check_token(self, token):
    return set(token) <= self.safe_characters and len(token) == 32