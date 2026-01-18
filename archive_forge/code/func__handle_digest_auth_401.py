from requests import auth
from requests import cookies
from . import _digest_auth_compat as auth_compat, http_proxy_digest
def _handle_digest_auth_401(self, r, kwargs):
    self.auth = auth_compat.HTTPDigestAuth(self.username, self.password)
    try:
        self.auth.init_per_thread_state()
    except AttributeError:
        pass
    if hasattr(self.auth, 'num_401_calls') and self.auth.num_401_calls is None:
        self.auth.num_401_calls = 1
    return self.auth.handle_401(r, **kwargs)