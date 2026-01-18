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
class TestImageMember(test_utils.BaseTestCase):

    def setUp(self):
        super(TestImageMember, self).setUp()
        self.image_member_factory = domain.ImageMemberFactory()
        self.image_factory = domain.ImageFactory()
        self.image = self.image_factory.new_image()
        self.image_member = self.image_member_factory.new_image_member(image=self.image, member_id=TENANT1)

    def test_status_enumerated(self):
        self.image_member.status = 'pending'
        self.image_member.status = 'accepted'
        self.image_member.status = 'rejected'
        self.assertRaises(ValueError, setattr, self.image_member, 'status', 'ellison')