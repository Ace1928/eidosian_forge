from troveclient.compat import exceptions
class Auth1_1(Authenticator):

    def authenticate(self):
        """Authenticate against a v2.0 auth service."""
        if self.url is None:
            raise exceptions.AuthUrlNotGiven()
        auth_url = self.url
        body = {'credentials': {'username': self.username, 'key': self.password}}
        return self._authenticate(auth_url, body, root_key='auth')