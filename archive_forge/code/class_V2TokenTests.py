import uuid
from keystoneauth1 import fixture
from keystoneauth1.tests.unit import utils
class V2TokenTests(utils.TestCase):

    def test_unscoped(self):
        token_id = uuid.uuid4().hex
        user_id = uuid.uuid4().hex
        user_name = uuid.uuid4().hex
        token = fixture.V2Token(token_id=token_id, user_id=user_id, user_name=user_name)
        self.assertEqual(token_id, token.token_id)
        self.assertEqual(token_id, token['access']['token']['id'])
        self.assertEqual(user_id, token.user_id)
        self.assertEqual(user_id, token['access']['user']['id'])
        self.assertEqual(user_name, token.user_name)
        self.assertEqual(user_name, token['access']['user']['name'])
        self.assertIsNone(token.trust_id)

    def test_tenant_scoped(self):
        tenant_id = uuid.uuid4().hex
        tenant_name = uuid.uuid4().hex
        token = fixture.V2Token(tenant_id=tenant_id, tenant_name=tenant_name)
        self.assertEqual(tenant_id, token.tenant_id)
        self.assertEqual(tenant_id, token['access']['token']['tenant']['id'])
        self.assertEqual(tenant_name, token.tenant_name)
        tn = token['access']['token']['tenant']['name']
        self.assertEqual(tenant_name, tn)
        self.assertIsNone(token.trust_id)

    def test_trust_scoped(self):
        trust_id = uuid.uuid4().hex
        trustee_user_id = uuid.uuid4().hex
        token = fixture.V2Token(trust_id=trust_id, trustee_user_id=trustee_user_id)
        trust = token['access']['trust']
        self.assertEqual(trust_id, token.trust_id)
        self.assertEqual(trust_id, trust['id'])
        self.assertEqual(trustee_user_id, token.trustee_user_id)
        self.assertEqual(trustee_user_id, trust['trustee_user_id'])

    def test_roles(self):
        role_id1 = uuid.uuid4().hex
        role_name1 = uuid.uuid4().hex
        role_id2 = uuid.uuid4().hex
        role_name2 = uuid.uuid4().hex
        token = fixture.V2Token()
        token.add_role(id=role_id1, name=role_name1)
        token.add_role(id=role_id2, name=role_name2)
        role_names = token['access']['user']['roles']
        role_ids = token['access']['metadata']['roles']
        self.assertEqual(set([role_id1, role_id2]), set(role_ids))
        for r in (role_name1, role_name2):
            self.assertIn({'name': r}, role_names)

    def test_services(self):
        service_type = uuid.uuid4().hex
        service_name = uuid.uuid4().hex
        endpoint_id = uuid.uuid4().hex
        region = uuid.uuid4().hex
        public = uuid.uuid4().hex
        admin = uuid.uuid4().hex
        internal = uuid.uuid4().hex
        token = fixture.V2Token()
        svc = token.add_service(type=service_type, name=service_name)
        svc.add_endpoint(public=public, admin=admin, internal=internal, region=region, id=endpoint_id)
        self.assertEqual(1, len(token['access']['serviceCatalog']))
        service = token['access']['serviceCatalog'][0]['endpoints'][0]
        self.assertEqual(public, service['publicURL'])
        self.assertEqual(internal, service['internalURL'])
        self.assertEqual(admin, service['adminURL'])
        self.assertEqual(region, service['region'])
        self.assertEqual(endpoint_id, service['id'])
        token.remove_service(type=service_type)
        self.assertEqual(0, len(token['access']['serviceCatalog']))

    def test_token_bind(self):
        name1 = uuid.uuid4().hex
        data1 = uuid.uuid4().hex
        name2 = uuid.uuid4().hex
        data2 = {uuid.uuid4().hex: uuid.uuid4().hex}
        token = fixture.V2Token()
        token.set_bind(name1, data1)
        token.set_bind(name2, data2)
        self.assertEqual({name1: data1, name2: data2}, token['access']['token']['bind'])