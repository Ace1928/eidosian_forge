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
class TestImage(test_utils.BaseTestCase):

    def setUp(self):
        super(TestImage, self).setUp()
        self.image_factory = domain.ImageFactory()
        self.image = self.image_factory.new_image(container_format='bear', disk_format='rawr')

    def test_extra_properties(self):
        self.image.extra_properties = {'foo': 'bar'}
        self.assertEqual({'foo': 'bar'}, self.image.extra_properties)

    def test_extra_properties_assign(self):
        self.image.extra_properties['foo'] = 'bar'
        self.assertEqual({'foo': 'bar'}, self.image.extra_properties)

    def test_delete_extra_properties(self):
        self.image.extra_properties = {'foo': 'bar'}
        self.assertEqual({'foo': 'bar'}, self.image.extra_properties)
        del self.image.extra_properties['foo']
        self.assertEqual({}, self.image.extra_properties)

    def test_visibility_enumerated(self):
        self.image.visibility = 'public'
        self.image.visibility = 'private'
        self.image.visibility = 'shared'
        self.image.visibility = 'community'
        self.assertRaises(ValueError, setattr, self.image, 'visibility', 'ellison')

    def test_tags_always_a_set(self):
        self.image.tags = ['a', 'b', 'c']
        self.assertEqual(set(['a', 'b', 'c']), self.image.tags)

    def test_delete_protected_image(self):
        self.image.protected = True
        self.assertRaises(exception.ProtectedImageDelete, self.image.delete)

    def test_status_saving(self):
        self.image.status = 'saving'
        self.assertEqual('saving', self.image.status)

    def test_set_incorrect_status(self):
        self.image.status = 'saving'
        self.image.status = 'killed'
        self.assertRaises(exception.InvalidImageStatusTransition, setattr, self.image, 'status', 'delet')

    def test_status_saving_without_disk_format(self):
        self.image.disk_format = None
        self.assertRaises(ValueError, setattr, self.image, 'status', 'saving')

    def test_status_saving_without_container_format(self):
        self.image.container_format = None
        self.assertRaises(ValueError, setattr, self.image, 'status', 'saving')

    def test_status_active_without_disk_format(self):
        self.image.disk_format = None
        self.assertRaises(ValueError, setattr, self.image, 'status', 'active')

    def test_status_active_without_container_format(self):
        self.image.container_format = None
        self.assertRaises(ValueError, setattr, self.image, 'status', 'active')

    def test_delayed_delete(self):
        self.config(delayed_delete=True)
        self.image.status = 'active'
        self.image.locations = [{'url': 'http://foo.bar/not.exists', 'metadata': {}}]
        self.assertEqual('active', self.image.status)
        self.image.delete()
        self.assertEqual('pending_delete', self.image.status)