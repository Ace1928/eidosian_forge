import uuid
from keystone.common import provider_api
from keystone import exception
from keystone.tests import unit
class PolicyTests(object):

    def test_create(self):
        ref = unit.new_policy_ref()
        res = PROVIDERS.policy_api.create_policy(ref['id'], ref)
        self.assertDictEqual(ref, res)

    def test_get(self):
        ref = unit.new_policy_ref()
        res = PROVIDERS.policy_api.create_policy(ref['id'], ref)
        res = PROVIDERS.policy_api.get_policy(ref['id'])
        self.assertDictEqual(ref, res)

    def test_list(self):
        ref = unit.new_policy_ref()
        PROVIDERS.policy_api.create_policy(ref['id'], ref)
        res = PROVIDERS.policy_api.list_policies()
        res = [x for x in res if x['id'] == ref['id']][0]
        self.assertDictEqual(ref, res)

    def test_update(self):
        ref = unit.new_policy_ref()
        PROVIDERS.policy_api.create_policy(ref['id'], ref)
        orig = ref
        ref = unit.new_policy_ref()
        self.assertRaises(exception.ValidationError, PROVIDERS.policy_api.update_policy, orig['id'], ref)
        ref['id'] = orig['id']
        res = PROVIDERS.policy_api.update_policy(orig['id'], ref)
        self.assertDictEqual(ref, res)

    def test_delete(self):
        ref = unit.new_policy_ref()
        PROVIDERS.policy_api.create_policy(ref['id'], ref)
        PROVIDERS.policy_api.delete_policy(ref['id'])
        self.assertRaises(exception.PolicyNotFound, PROVIDERS.policy_api.delete_policy, ref['id'])
        self.assertRaises(exception.PolicyNotFound, PROVIDERS.policy_api.get_policy, ref['id'])
        res = PROVIDERS.policy_api.list_policies()
        self.assertFalse(len([x for x in res if x['id'] == ref['id']]))

    def test_get_policy_returns_not_found(self):
        self.assertRaises(exception.PolicyNotFound, PROVIDERS.policy_api.get_policy, uuid.uuid4().hex)

    def test_update_policy_returns_not_found(self):
        ref = unit.new_policy_ref()
        self.assertRaises(exception.PolicyNotFound, PROVIDERS.policy_api.update_policy, ref['id'], ref)

    def test_delete_policy_returns_not_found(self):
        self.assertRaises(exception.PolicyNotFound, PROVIDERS.policy_api.delete_policy, uuid.uuid4().hex)