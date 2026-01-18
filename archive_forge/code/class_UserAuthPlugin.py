from keystoneauth1.identity import base as base_identity
class UserAuthPlugin(base_identity.BaseIdentityPlugin):
    """The incoming authentication credentials.

    A plugin that represents the incoming user credentials. This can be
    consumed by applications.

    This object is not expected to be constructed directly by users. It is
    created and passed by auth_token middleware and then can be used as the
    authentication plugin when communicating via a session.
    """

    def __init__(self, user_auth_ref, serv_auth_ref, session=None, auth=None):
        super(UserAuthPlugin, self).__init__(reauthenticate=False)
        self.user = user_auth_ref
        self.service = serv_auth_ref
        self._session = session
        self._auth = auth

    @property
    def has_user_token(self):
        """Did this authentication request contained a user auth token."""
        return self.user is not None

    @property
    def has_service_token(self):
        """Did this authentication request contained a service token."""
        return self.service is not None

    def get_auth_ref(self, session, **kwargs):
        return self.user

    @property
    def _log_format(self):
        msg = []
        if self.has_user_token:
            msg.append('user: %s' % _log_format(self.user))
        if self.has_service_token:
            msg.append('service: %s' % _log_format(self.service))
        return ' '.join(msg)

    def get_headers(self, session, **kwargs):
        headers = super(UserAuthPlugin, self).get_headers(session, **kwargs)
        if headers is not None and self._session:
            token = self._session.get_token(auth=self._auth)
            if token:
                headers['X-Service-Token'] = token
        return headers