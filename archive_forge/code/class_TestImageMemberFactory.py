import datetime
from unittest import mock
import uuid
from oslo_config import cfg
import oslo_utils.importutils
import glance.async_
from glance.async_ import taskflow_executor
from glance.common import exception
from glance.common import timeutils
from glance import domain
import glance.tests.utils as test_utils
class TestImageMemberFactory(test_utils.BaseTestCase):

    def setUp(self):
        super(TestImageMemberFactory, self).setUp()
        self.image_member_factory = domain.ImageMemberFactory()
        self.image_factory = domain.ImageFactory()

    def test_minimal_new_image_member(self):
        member_id = 'fake-member-id'
        image = self.image_factory.new_image(image_id=UUID1, name='image-1', min_disk=256, owner=TENANT1)
        image_member = self.image_member_factory.new_image_member(image, member_id)
        self.assertEqual(image_member.image_id, image.image_id)
        self.assertIsNotNone(image_member.created_at)
        self.assertEqual(image_member.created_at, image_member.updated_at)
        self.assertEqual('pending', image_member.status)
        self.assertIsNotNone(image_member.member_id)