import abc
class NoAuth(AuthBackend):
    """No Auth Plugin."""

    def authenticate(self, api_version, req):
        return req