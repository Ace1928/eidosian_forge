import uuid
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import groups
class GroupTests(utils.ClientTestCase, utils.CrudTests):

    def setUp(self):
        super(GroupTests, self).setUp()
        self.key = 'group'
        self.collection_key = 'groups'
        self.model = groups.Group
        self.manager = self.client.groups

    def new_ref(self, **kwargs):
        kwargs = super(GroupTests, self).new_ref(**kwargs)
        kwargs.setdefault('name', uuid.uuid4().hex)
        return kwargs

    def test_list_groups_for_user(self):
        user_id = uuid.uuid4().hex
        ref_list = [self.new_ref(), self.new_ref()]
        self.stub_entity('GET', ['users', user_id, self.collection_key], status_code=200, entity=ref_list)
        returned_list = self.manager.list(user=user_id)
        self.assertEqual(len(ref_list), len(returned_list))
        [self.assertIsInstance(r, self.model) for r in returned_list]

    def test_list_groups_for_domain(self):
        ref_list = [self.new_ref(), self.new_ref()]
        domain_id = uuid.uuid4().hex
        self.stub_entity('GET', [self.collection_key], status_code=200, entity=ref_list)
        returned_list = self.manager.list(domain=domain_id)
        self.assertTrue(len(ref_list), len(returned_list))
        [self.assertIsInstance(r, self.model) for r in returned_list]
        self.assertQueryStringIs('domain_id=%s' % domain_id)