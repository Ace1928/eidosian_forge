from keystoneauth1 import exceptions
from keystoneauth1.identity import v3
class _Saml2TokenAuthMethod(v3.AuthMethod):
    _method_parameters = []

    def get_auth_data(self, session, auth, headers, **kwargs):
        raise exceptions.MethodNotImplemented('This method should never be called')