from keystoneclient.i18n import _
from keystoneclient.v3.contrib.oauth1 import access_tokens
from keystoneclient.v3.contrib.oauth1 import consumers
from keystoneclient.v3.contrib.oauth1 import request_tokens
class OAuthManager(object):

    def __init__(self, api):
        self.access_tokens = access_tokens.AccessTokenManager(api)
        self.consumers = consumers.ConsumerManager(api)
        self.request_tokens = request_tokens.RequestTokenManager(api)