from troveclient.compat import exceptions
class KeyStoneV2Authenticator(Authenticator):

    def authenticate(self):
        if self.url is None:
            raise exceptions.AuthUrlNotGiven()
        return self._v2_auth(self.url)

    def _v2_auth(self, url):
        """Authenticate against a v2.0 auth service."""
        body = {'auth': {'passwordCredentials': {'username': self.username, 'password': self.password}}}
        if self.tenant:
            body['auth']['tenantName'] = self.tenant
        return self._authenticate(url, body)