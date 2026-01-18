import datetime
import uuid
from keystoneauth1 import _utils
from keystoneauth1.fixture import exception
class V3FederationToken(Token):
    """A V3 Keystone Federation token that can be used for testing.

    Similar to V3Token, this object is designed to allow clients to generate
    a correct V3 federation token for use in test code.
    """
    FEDERATED_DOMAIN_ID = 'Federated'

    def __init__(self, methods=None, identity_provider=None, protocol=None, groups=None):
        methods = methods or ['saml2']
        super(V3FederationToken, self).__init__(methods=methods)
        self._user_domain = {'id': V3FederationToken.FEDERATED_DOMAIN_ID}
        self.add_federation_info_to_user(identity_provider, protocol, groups)

    def add_federation_info_to_user(self, identity_provider=None, protocol=None, groups=None):
        data = {'OS-FEDERATION': {'identity_provider': identity_provider or uuid.uuid4().hex, 'protocol': protocol or uuid.uuid4().hex, 'groups': groups or [{'id': uuid.uuid4().hex}]}}
        self._user.update(data)
        return data