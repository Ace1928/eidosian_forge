from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from boto.auth_handler import AuthHandler
class NoOpAuth(AuthHandler):
    """No-op authorization plugin class."""
    capability = ['hmac-v4-s3', 's3']

    def __init__(self, path, config, provider):
        pass

    def add_auth(self, http_request):
        pass