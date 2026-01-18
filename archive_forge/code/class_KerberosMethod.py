from keystoneauth1 import access
from keystoneauth1.identity import v3
from keystoneauth1.identity.v3 import federation
class KerberosMethod(v3.AuthMethod):
    _method_parameters = ['mutual_auth']

    def __init__(self, *args, **kwargs):
        _dependency_check()
        super(KerberosMethod, self).__init__(*args, **kwargs)

    def get_auth_data(self, session, auth, headers, request_kwargs, **kwargs):
        request_kwargs['requests_auth'] = _requests_auth(self.mutual_auth)
        return ('kerberos', {})