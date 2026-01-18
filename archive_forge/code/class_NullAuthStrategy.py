from requests.auth import AuthBase, HTTPBasicAuth
from requests.compat import urlparse, urlunparse
class NullAuthStrategy(AuthBase):

    def __repr__(self):
        return '<NullAuthStrategy>'

    def __call__(self, r):
        return r