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
class TestImageTagQuotas(test_utils.BaseTestCase):

    def setUp(self):
        super(TestImageTagQuotas, self).setUp()
        self.base_image = mock.Mock()
        self.base_image.tags = set([])
        self.base_image.extra_properties = {}
        self.image = glance.quota.ImageProxy(self.base_image, mock.Mock(), mock.Mock(), mock.Mock())
        self.image_repo_mock = mock.Mock()
        self.image_repo_proxy = glance.quota.ImageRepoProxy(self.image_repo_mock, mock.Mock(), mock.Mock(), mock.Mock())

    def test_replace_image_tag(self):
        self.config(image_tag_quota=1)
        self.image.tags = ['foo']
        self.assertEqual(1, len(self.image.tags))

    def test_replace_too_many_image_tags(self):
        self.config(image_tag_quota=0)
        exc = self.assertRaises(exception.ImageTagLimitExceeded, setattr, self.image, 'tags', ['foo', 'bar'])
        self.assertIn('Attempted: 2, Maximum: 0', encodeutils.exception_to_unicode(exc))
        self.assertEqual(0, len(self.image.tags))

    def test_replace_unlimited_image_tags(self):
        self.config(image_tag_quota=-1)
        self.image.tags = ['foo']
        self.assertEqual(1, len(self.image.tags))

    def test_add_image_tag(self):
        self.config(image_tag_quota=1)
        self.image.tags.add('foo')
        self.assertEqual(1, len(self.image.tags))

    def test_add_too_many_image_tags(self):
        self.config(image_tag_quota=1)
        self.image.tags.add('foo')
        exc = self.assertRaises(exception.ImageTagLimitExceeded, self.image.tags.add, 'bar')
        self.assertIn('Attempted: 2, Maximum: 1', encodeutils.exception_to_unicode(exc))

    def test_add_unlimited_image_tags(self):
        self.config(image_tag_quota=-1)
        self.image.tags.add('foo')
        self.assertEqual(1, len(self.image.tags))

    def test_remove_image_tag_while_over_quota(self):
        self.config(image_tag_quota=1)
        self.image.tags.add('foo')
        self.assertEqual(1, len(self.image.tags))
        self.config(image_tag_quota=0)
        self.image.tags.remove('foo')
        self.assertEqual(0, len(self.image.tags))