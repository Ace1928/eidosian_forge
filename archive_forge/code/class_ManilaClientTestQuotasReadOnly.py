from random import randint
import ddt
from tempest.lib.cli import output_parser
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from manilaclient import api_versions
from manilaclient.tests.functional import base
from manilaclient.tests.functional import utils
@ddt.ddt
class ManilaClientTestQuotasReadOnly(base.BaseTestCase):

    def test_quota_class_show_by_admin(self):
        roles = self.parser.listing(self.clients['admin'].manila('quota-class-show', params='abc'))
        self.assertTableStruct(roles, ('Property', 'Value'))

    def test_quota_class_show_by_user(self):
        self.assertRaises(exceptions.CommandFailed, self.clients['user'].manila, 'quota-class-show', params='abc')

    def _get_quotas(self, role, operation, microversion):
        roles = self.parser.listing(self.clients[role].manila('quota-%s' % operation, microversion=microversion))
        self.assertTableStruct(roles, ('Property', 'Value'))

    @ddt.data('admin', 'user')
    @utils.skip_if_microversion_not_supported('1.0')
    def test_quota_defaults_api_1_0(self, role):
        self._get_quotas(role, 'defaults', '1.0')

    @ddt.data('admin', 'user')
    @utils.skip_if_microversion_not_supported('2.0')
    def test_quota_defaults_api_2_0(self, role):
        self._get_quotas(role, 'defaults', '2.0')

    @ddt.data('admin', 'user')
    @utils.skip_if_microversion_not_supported('2.6')
    def test_quota_defaults_api_2_6(self, role):
        self._get_quotas(role, 'defaults', '2.6')

    @ddt.data('admin', 'user')
    @utils.skip_if_microversion_not_supported('2.7')
    def test_quota_defaults_api_2_7(self, role):
        self._get_quotas(role, 'defaults', '2.7')

    @ddt.data('admin', 'user')
    @utils.skip_if_microversion_not_supported('1.0')
    def test_quota_show_api_1_0(self, role):
        self._get_quotas(role, 'show', '1.0')

    @ddt.data('admin', 'user')
    @utils.skip_if_microversion_not_supported('2.0')
    def test_quota_show_api_2_0(self, role):
        self._get_quotas(role, 'show', '2.0')

    @ddt.data('admin', 'user')
    @utils.skip_if_microversion_not_supported('2.6')
    def test_quota_show_api_2_6(self, role):
        self._get_quotas(role, 'show', '2.6')

    @ddt.data('admin', 'user')
    @utils.skip_if_microversion_not_supported('2.7')
    def test_quota_show_api_2_7(self, role):
        self._get_quotas(role, 'show', '2.7')

    @ddt.data('admin', 'user')
    @utils.skip_if_microversion_not_supported('2.25')
    def test_quota_show_api_2_25(self, role):
        self._get_quotas(role, 'show  --detail', '2.25')