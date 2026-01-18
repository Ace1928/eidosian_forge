import ddt
from manilaclient.tests.functional import base
@ddt.ddt
class ManilaClientTestServicesReadOnly(base.BaseTestCase):

    @ddt.data('1.0', '2.0', '2.6', '2.7')
    def test_services_list(self, microversion):
        self.skip_if_microversion_not_supported(microversion)
        self.admin_client.manila('service-list', microversion=microversion)

    def test_list_with_debug_flag(self):
        self.clients['admin'].manila('service-list', flags='--debug')

    def test_shares_list_filter_by_host(self):
        self.clients['admin'].manila('service-list', params='--host host')

    def test_shares_list_filter_by_binary(self):
        self.clients['admin'].manila('service-list', params='--binary binary')

    def test_shares_list_filter_by_zone(self):
        self.clients['admin'].manila('service-list', params='--zone zone')

    def test_shares_list_filter_by_status(self):
        self.clients['admin'].manila('service-list', params='--status status')

    def test_shares_list_filter_by_state(self):
        self.clients['admin'].manila('service-list', params='--state state')