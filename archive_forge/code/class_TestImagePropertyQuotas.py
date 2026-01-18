import copy
import fixtures
from unittest import mock
from unittest.mock import patch
import uuid
from oslo_limit import exception as ol_exc
from oslo_utils import encodeutils
from oslo_utils import units
from glance.common import exception
from glance.common import store_utils
import glance.quota
from glance.quota import keystone as ks_quota
from glance.tests.unit import fixtures as glance_fixtures
from glance.tests.unit import utils as unit_test_utils
from glance.tests import utils as test_utils
class TestImagePropertyQuotas(test_utils.BaseTestCase):

    def setUp(self):
        super(TestImagePropertyQuotas, self).setUp()
        self.base_image = FakeImage()
        self.image = glance.quota.ImageProxy(self.base_image, mock.Mock(), mock.Mock(), mock.Mock())
        self.image_repo_mock = mock.Mock()
        self.image_repo_mock.add.return_value = self.base_image
        self.image_repo_mock.save.return_value = self.base_image
        self.image_repo_proxy = glance.quota.ImageRepoProxy(self.image_repo_mock, mock.Mock(), mock.Mock(), mock.Mock())

    def test_save_image_with_image_property(self):
        self.config(image_property_quota=1)
        self.image.extra_properties = {'foo': 'bar'}
        self.image_repo_proxy.save(self.image)
        self.image_repo_mock.save.assert_called_once_with(self.base_image, from_state=None)

    def test_save_image_too_many_image_properties(self):
        self.config(image_property_quota=1)
        self.image.extra_properties = {'foo': 'bar', 'foo2': 'bar2'}
        exc = self.assertRaises(exception.ImagePropertyLimitExceeded, self.image_repo_proxy.save, self.image)
        self.assertIn('Attempted: 2, Maximum: 1', encodeutils.exception_to_unicode(exc))

    def test_save_image_unlimited_image_properties(self):
        self.config(image_property_quota=-1)
        self.image.extra_properties = {'foo': 'bar'}
        self.image_repo_proxy.save(self.image)
        self.image_repo_mock.save.assert_called_once_with(self.base_image, from_state=None)

    def test_add_image_with_image_property(self):
        self.config(image_property_quota=1)
        self.image.extra_properties = {'foo': 'bar'}
        self.image_repo_proxy.add(self.image)
        self.image_repo_mock.add.assert_called_once_with(self.base_image)

    def test_add_image_too_many_image_properties(self):
        self.config(image_property_quota=1)
        self.image.extra_properties = {'foo': 'bar', 'foo2': 'bar2'}
        exc = self.assertRaises(exception.ImagePropertyLimitExceeded, self.image_repo_proxy.add, self.image)
        self.assertIn('Attempted: 2, Maximum: 1', encodeutils.exception_to_unicode(exc))

    def test_add_image_unlimited_image_properties(self):
        self.config(image_property_quota=-1)
        self.image.extra_properties = {'foo': 'bar'}
        self.image_repo_proxy.add(self.image)
        self.image_repo_mock.add.assert_called_once_with(self.base_image)

    def _quota_exceed_setup(self):
        self.config(image_property_quota=2)
        self.base_image.extra_properties = {'foo': 'bar', 'spam': 'ham'}
        self.image = glance.quota.ImageProxy(self.base_image, mock.Mock(), mock.Mock(), mock.Mock())

    def test_modify_image_properties_when_quota_exceeded(self):
        self._quota_exceed_setup()
        self.config(image_property_quota=1)
        self.image.extra_properties = {'foo': 'frob', 'spam': 'eggs'}
        self.image_repo_proxy.save(self.image)
        self.image_repo_mock.save.assert_called_once_with(self.base_image, from_state=None)
        self.assertEqual('frob', self.base_image.extra_properties['foo'])
        self.assertEqual('eggs', self.base_image.extra_properties['spam'])

    def test_delete_image_properties_when_quota_exceeded(self):
        self._quota_exceed_setup()
        self.config(image_property_quota=1)
        del self.image.extra_properties['foo']
        self.image_repo_proxy.save(self.image)
        self.image_repo_mock.save.assert_called_once_with(self.base_image, from_state=None)
        self.assertNotIn('foo', self.base_image.extra_properties)
        self.assertEqual('ham', self.base_image.extra_properties['spam'])

    def test_invalid_quota_config_parameter(self):
        self.config(user_storage_quota='foo')
        location = {'url': 'file:///fake.img.tar.gz', 'metadata': {}}
        self.assertRaises(exception.InvalidOptionValue, self.image.locations.append, location)

    def test_exceed_quota_during_patch_operation(self):
        self._quota_exceed_setup()
        self.image.extra_properties['frob'] = 'baz'
        self.image.extra_properties['lorem'] = 'ipsum'
        self.assertEqual('bar', self.base_image.extra_properties['foo'])
        self.assertEqual('ham', self.base_image.extra_properties['spam'])
        self.assertEqual('baz', self.base_image.extra_properties['frob'])
        self.assertEqual('ipsum', self.base_image.extra_properties['lorem'])
        del self.image.extra_properties['frob']
        del self.image.extra_properties['lorem']
        self.image_repo_proxy.save(self.image)
        call_args = mock.call(self.base_image, from_state=None)
        self.assertEqual(call_args, self.image_repo_mock.save.call_args)
        self.assertEqual('bar', self.base_image.extra_properties['foo'])
        self.assertEqual('ham', self.base_image.extra_properties['spam'])
        self.assertNotIn('frob', self.base_image.extra_properties)
        self.assertNotIn('lorem', self.base_image.extra_properties)

    def test_quota_exceeded_after_delete_image_properties(self):
        self.config(image_property_quota=3)
        self.base_image.extra_properties = {'foo': 'bar', 'spam': 'ham', 'frob': 'baz'}
        self.image = glance.quota.ImageProxy(self.base_image, mock.Mock(), mock.Mock(), mock.Mock())
        self.config(image_property_quota=1)
        del self.image.extra_properties['foo']
        self.image_repo_proxy.save(self.image)
        self.image_repo_mock.save.assert_called_once_with(self.base_image, from_state=None)
        self.assertNotIn('foo', self.base_image.extra_properties)
        self.assertEqual('ham', self.base_image.extra_properties['spam'])
        self.assertEqual('baz', self.base_image.extra_properties['frob'])