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
class TestImageKeystoneQuota(test_utils.BaseTestCase):

    def setUp(self):
        super(TestImageKeystoneQuota, self).setUp()
        default_limits = {ks_quota.QUOTA_IMAGE_SIZE_TOTAL: 500, 'another_limit': 2}
        ksqf = glance_fixtures.KeystoneQuotaFixture(**default_limits)
        self.useFixture(ksqf)
        self.db_api = unit_test_utils.FakeDB()
        self.useFixture(fixtures.MockPatch('glance.quota.keystone.db', self.db_api))

    def _create_fake_image(self, context, size):
        location_count = 2
        locations = []
        for i in range(location_count):
            locations.append({'url': 'file:///g/there/it/is%d' % i, 'status': 'active', 'metadata': {}})
        image_values = {'id': str(uuid.uuid4()), 'owner': context.owner, 'status': 'active', 'size': size * units.Mi, 'locations': locations}
        self.db_api.image_create(context, image_values)

    def test_enforce_overquota(self):
        self.config(endpoint_id='ENDPOINT_ID', group='oslo_limit')
        self.config(use_keystone_limits=True)
        context = FakeContext()
        self._create_fake_image(context, 300)
        exc = self.assertRaises(exception.LimitExceeded, ks_quota.enforce_image_size_total, context, context.owner)
        self.assertIn('image_size_total is over limit of 500', str(exc))

    def test_enforce_overquota_with_delta(self):
        self.config(endpoint_id='ENDPOINT_ID', group='oslo_limit')
        self.config(use_keystone_limits=True)
        context = FakeContext()
        self._create_fake_image(context, 200)
        ks_quota.enforce_image_size_total(context, context.owner)
        ks_quota.enforce_image_size_total(context, context.owner, delta=50)
        self.assertRaises(exception.LimitExceeded, ks_quota.enforce_image_size_total, context, context.owner, delta=200)

    def test_enforce_overquota_disabled(self):
        self.config(endpoint_id='ENDPOINT_ID', group='oslo_limit')
        self.config(use_keystone_limits=False)
        context = FakeContext()
        self._create_fake_image(context, 300)
        ks_quota.enforce_image_size_total(context, context.owner)

    def test_enforce_overquota_multiple(self):
        self.config(endpoint_id='ENDPOINT_ID', group='oslo_limit')
        self.config(use_keystone_limits=True)
        context = FakeContext()
        self._create_fake_image(context, 150)
        self._create_fake_image(context, 150)
        exc = self.assertRaises(exception.LimitExceeded, ks_quota.enforce_image_size_total, context, context.owner)
        self.assertIn('image_size_total is over limit of 500', str(exc))

    def test_enforce_underquota(self):
        self.config(endpoint_id='ENDPOINT_ID', group='oslo_limit')
        self.config(use_keystone_limits=True)
        context = FakeContext()
        self._create_fake_image(context, 100)
        ks_quota.enforce_image_size_total(context, context.owner)

    def test_enforce_underquota_with_others_over_quota(self):
        self.config(endpoint_id='ENDPOINT_ID', group='oslo_limit')
        self.config(use_keystone_limits=True)
        context = FakeContext()
        self._create_fake_image(context, 300)
        self._create_fake_image(context, 300)
        other_context = FakeContext()
        other_context.owner = 'someone_else'
        self._create_fake_image(other_context, 100)
        ks_quota.enforce_image_size_total(other_context, other_context.owner)

    def test_enforce_multiple_limits_under_quota(self):
        self.config(endpoint_id='ENDPOINT_ID', group='oslo_limit')
        self.config(use_keystone_limits=True)
        context = FakeContext()
        ks_quota._enforce_some(context, context.owner, {ks_quota.QUOTA_IMAGE_SIZE_TOTAL: lambda: 200, 'another_limit': lambda: 1}, {'another_limit': 1})

    def test_enforce_multiple_limits_over_quota(self):
        self.config(endpoint_id='ENDPOINT_ID', group='oslo_limit')
        self.config(use_keystone_limits=True)
        context = FakeContext()
        self.assertRaises(exception.LimitExceeded, ks_quota._enforce_some, context, context.owner, {ks_quota.QUOTA_IMAGE_SIZE_TOTAL: lambda: 200, 'another_limit': lambda: 1}, {'another_limit': 5})

    @mock.patch('oslo_limit.limit.Enforcer')
    @mock.patch.object(ks_quota, 'LOG')
    def test_oslo_limit_config_fail(self, mock_LOG, mock_enforcer):
        self.config(endpoint_id='ENDPOINT_ID', group='oslo_limit')
        self.config(use_keystone_limits=True)
        mock_enforcer.return_value.enforce.side_effect = ol_exc.SessionInitError('test')
        context = FakeContext()
        self._create_fake_image(context, 100)
        self.assertRaises(ol_exc.SessionInitError, ks_quota.enforce_image_size_total, context, context.owner)
        mock_LOG.error.assert_called_once_with('Failed to initialize oslo_limit, likely due to incorrect or insufficient configuration: %(err)s', {'err': "Can't initialise OpenStackSDK session: test."})