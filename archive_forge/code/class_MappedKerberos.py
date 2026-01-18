from keystoneauth1 import access
from keystoneauth1.identity import v3
from keystoneauth1.identity.v3 import federation
class MappedKerberos(federation.FederationBaseAuth):
    """Authenticate using Kerberos via the keystone federation mechanisms.

    This uses the OS-FEDERATION extension to gain an unscoped token and then
    use the standard keystone auth process to scope that to any given project.
    """

    def __init__(self, auth_url, identity_provider, protocol, mutual_auth=None, **kwargs):
        _dependency_check()
        self.mutual_auth = mutual_auth
        super(MappedKerberos, self).__init__(auth_url, identity_provider, protocol, **kwargs)

    def get_unscoped_auth_ref(self, session, **kwargs):
        resp = session.get(self.federated_token_url, requests_auth=_requests_auth(self.mutual_auth), authenticated=False)
        return access.create(body=resp.json(), resp=resp)