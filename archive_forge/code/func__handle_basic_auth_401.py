from requests import auth
from requests import cookies
from . import _digest_auth_compat as auth_compat, http_proxy_digest
def _handle_basic_auth_401(self, r, kwargs):
    if self.pos is not None:
        r.request.body.seek(self.pos)
    r.content
    r.raw.release_conn()
    prep = r.request.copy()
    if not hasattr(prep, '_cookies'):
        prep._cookies = cookies.RequestsCookieJar()
    cookies.extract_cookies_to_jar(prep._cookies, r.request, r.raw)
    prep.prepare_cookies(prep._cookies)
    self.auth = auth.HTTPBasicAuth(self.username, self.password)
    prep = self.auth(prep)
    _r = r.connection.send(prep, **kwargs)
    _r.history.append(r)
    _r.request = prep
    return _r