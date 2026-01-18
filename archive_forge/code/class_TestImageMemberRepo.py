import datetime
from unittest import mock
import uuid
from oslo_config import cfg
from oslo_db import exception as db_exc
from oslo_utils import encodeutils
from oslo_utils.fixture import uuidsentinel as uuids
from oslo_utils import timeutils
from sqlalchemy import orm as sa_orm
from glance.common import crypt
from glance.common import exception
import glance.context
import glance.db
from glance.db.sqlalchemy import api
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
class TestImageMemberRepo(test_utils.BaseTestCase):

    def setUp(self):
        super(TestImageMemberRepo, self).setUp()
        self.db = unit_test_utils.FakeDB(initialize=False)
        self.context = glance.context.RequestContext(user=USER1, tenant=TENANT1)
        self.image_repo = glance.db.ImageRepo(self.context, self.db)
        self.image_member_factory = glance.domain.ImageMemberFactory()
        self._create_images()
        self._create_image_members()
        image = self.image_repo.get(UUID1)
        self.image_member_repo = glance.db.ImageMemberRepo(self.context, self.db, image)

    def _create_images(self):
        self.images = [_db_fixture(UUID1, owner=TENANT1, name='1', size=256, status='active'), _db_fixture(UUID2, owner=TENANT1, name='2', size=512, visibility='shared')]
        [self.db.image_create(None, image) for image in self.images]
        self.db.image_tag_set_all(None, UUID1, ['ping', 'pong'])

    def _create_image_members(self):
        self.image_members = [_db_image_member_fixture(UUID1, TENANT2), _db_image_member_fixture(UUID1, TENANT3)]
        [self.db.image_member_create(None, image_member) for image_member in self.image_members]

    def test_list(self):
        image_members = self.image_member_repo.list()
        image_member_ids = set([i.member_id for i in image_members])
        self.assertEqual(set([TENANT2, TENANT3]), image_member_ids)

    def test_list_no_members(self):
        image = self.image_repo.get(UUID2)
        self.image_member_repo_uuid2 = glance.db.ImageMemberRepo(self.context, self.db, image)
        image_members = self.image_member_repo_uuid2.list()
        image_member_ids = set([i.member_id for i in image_members])
        self.assertEqual(set([]), image_member_ids)

    def test_save_image_member(self):
        image_member = self.image_member_repo.get(TENANT2)
        image_member.status = 'accepted'
        self.image_member_repo.save(image_member)
        image_member_updated = self.image_member_repo.get(TENANT2)
        self.assertEqual(image_member.id, image_member_updated.id)
        self.assertEqual('accepted', image_member_updated.status)

    def test_add_image_member(self):
        image = self.image_repo.get(UUID1)
        image_member = self.image_member_factory.new_image_member(image, TENANT4)
        self.assertIsNone(image_member.id)
        self.image_member_repo.add(image_member)
        retreived_image_member = self.image_member_repo.get(TENANT4)
        self.assertIsNotNone(retreived_image_member.id)
        self.assertEqual(image_member.image_id, retreived_image_member.image_id)
        self.assertEqual(image_member.member_id, retreived_image_member.member_id)
        self.assertEqual('pending', retreived_image_member.status)

    def test_add_duplicate_image_member(self):
        image = self.image_repo.get(UUID1)
        image_member = self.image_member_factory.new_image_member(image, TENANT4)
        self.assertIsNone(image_member.id)
        self.image_member_repo.add(image_member)
        retreived_image_member = self.image_member_repo.get(TENANT4)
        self.assertIsNotNone(retreived_image_member.id)
        self.assertEqual(image_member.image_id, retreived_image_member.image_id)
        self.assertEqual(image_member.member_id, retreived_image_member.member_id)
        self.assertEqual('pending', retreived_image_member.status)
        self.assertRaises(exception.Duplicate, self.image_member_repo.add, image_member)

    def test_get_image_member(self):
        image = self.image_repo.get(UUID1)
        image_member = self.image_member_factory.new_image_member(image, TENANT4)
        self.assertIsNone(image_member.id)
        self.image_member_repo.add(image_member)
        member = self.image_member_repo.get(image_member.member_id)
        self.assertEqual(member.id, image_member.id)
        self.assertEqual(member.image_id, image_member.image_id)
        self.assertEqual(member.member_id, image_member.member_id)
        self.assertEqual('pending', member.status)

    def test_get_nonexistent_image_member(self):
        fake_image_member_id = 'fake'
        self.assertRaises(exception.NotFound, self.image_member_repo.get, fake_image_member_id)

    def test_remove_image_member(self):
        image_member = self.image_member_repo.get(TENANT2)
        self.image_member_repo.remove(image_member)
        self.assertRaises(exception.NotFound, self.image_member_repo.get, TENANT2)

    def test_remove_image_member_does_not_exist(self):
        fake_uuid = str(uuid.uuid4())
        image = self.image_repo.get(UUID2)
        fake_member = glance.domain.ImageMemberFactory().new_image_member(image, TENANT4)
        fake_member.id = fake_uuid
        exc = self.assertRaises(exception.NotFound, self.image_member_repo.remove, fake_member)
        self.assertIn(fake_uuid, encodeutils.exception_to_unicode(exc))