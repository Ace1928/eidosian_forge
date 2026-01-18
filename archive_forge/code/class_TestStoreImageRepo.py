from cryptography import exceptions as crypto_exception
from cursive import exception as cursive_exception
from cursive import signature_utils
import glance_store
from unittest import mock
from glance.common import exception
import glance.location
from glance.tests.unit import base as unit_test_base
from glance.tests.unit import utils as unit_test_utils
from glance.tests import utils
class TestStoreImageRepo(utils.BaseTestCase):

    def setUp(self):
        super(TestStoreImageRepo, self).setUp()
        self.store_api = unit_test_utils.FakeStoreAPI()
        store_utils = unit_test_utils.FakeStoreUtils(self.store_api)
        self.image_stub = ImageStub(UUID1)
        self.image = glance.location.ImageProxy(self.image_stub, {}, self.store_api, store_utils)
        self.image_repo_stub = ImageRepoStub()
        self.image_repo = glance.location.ImageRepoProxy(self.image_repo_stub, {}, self.store_api, store_utils)
        patcher = mock.patch('glance.location._get_member_repo_for_store', self.get_fake_member_repo)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.fake_member_repo = FakeMemberRepo(self.image, [TENANT1, TENANT2])
        self.image_member_repo = glance.location.ImageMemberRepoProxy(self.fake_member_repo, self.image, {}, self.store_api)

    def get_fake_member_repo(self, image, context, db_api, store_api):
        return FakeMemberRepo(self.image, [TENANT1, TENANT2])

    def test_add_updates_acls(self):
        self.image_stub.locations = [{'url': 'foo', 'metadata': {}, 'status': 'active'}, {'url': 'bar', 'metadata': {}, 'status': 'active'}]
        self.image_stub.visibility = 'public'
        self.image_repo.add(self.image)
        self.assertTrue(self.store_api.acls['foo']['public'])
        self.assertEqual([], self.store_api.acls['foo']['read'])
        self.assertEqual([], self.store_api.acls['foo']['write'])
        self.assertTrue(self.store_api.acls['bar']['public'])
        self.assertEqual([], self.store_api.acls['bar']['read'])
        self.assertEqual([], self.store_api.acls['bar']['write'])

    def test_add_ignores_acls_if_no_locations(self):
        self.image_stub.locations = []
        self.image_stub.visibility = 'public'
        self.image_repo.add(self.image)
        self.assertEqual(0, len(self.store_api.acls))

    def test_save_updates_acls(self):
        self.image_stub.locations = [{'url': 'foo', 'metadata': {}, 'status': 'active'}]
        self.image_repo.save(self.image)
        self.assertIn('foo', self.store_api.acls)

    def test_add_fetches_members_if_private(self):
        self.image_stub.locations = [{'url': 'glue', 'metadata': {}, 'status': 'active'}]
        self.image_stub.visibility = 'private'
        self.image_repo.add(self.image)
        self.assertIn('glue', self.store_api.acls)
        acls = self.store_api.acls['glue']
        self.assertFalse(acls['public'])
        self.assertEqual([], acls['write'])
        self.assertEqual([TENANT1, TENANT2], acls['read'])

    def test_save_fetches_members_if_private(self):
        self.image_stub.locations = [{'url': 'glue', 'metadata': {}, 'status': 'active'}]
        self.image_stub.visibility = 'private'
        self.image_repo.save(self.image)
        self.assertIn('glue', self.store_api.acls)
        acls = self.store_api.acls['glue']
        self.assertFalse(acls['public'])
        self.assertEqual([], acls['write'])
        self.assertEqual([TENANT1, TENANT2], acls['read'])

    def test_member_addition_updates_acls(self):
        self.image_stub.locations = [{'url': 'glug', 'metadata': {}, 'status': 'active'}]
        self.image_stub.visibility = 'private'
        membership = glance.domain.ImageMembership(UUID1, TENANT3, None, None, status='accepted')
        self.image_member_repo.add(membership)
        self.assertIn('glug', self.store_api.acls)
        acls = self.store_api.acls['glug']
        self.assertFalse(acls['public'])
        self.assertEqual([], acls['write'])
        self.assertEqual([TENANT1, TENANT2, TENANT3], acls['read'])

    def test_member_removal_updates_acls(self):
        self.image_stub.locations = [{'url': 'glug', 'metadata': {}, 'status': 'active'}]
        self.image_stub.visibility = 'private'
        membership = glance.domain.ImageMembership(UUID1, TENANT1, None, None, status='accepted')
        self.image_member_repo.remove(membership)
        self.assertIn('glug', self.store_api.acls)
        acls = self.store_api.acls['glug']
        self.assertFalse(acls['public'])
        self.assertEqual([], acls['write'])
        self.assertEqual([TENANT2], acls['read'])