import ddt
from manilaclient.tests.functional import base
@ddt.ddt
class ManilaClientTestLimitsReadOnly(base.BaseTestCase):

    @ddt.data('admin', 'user')
    def test_rate_limits(self, role):
        self.clients[role].manila('rate-limits')

    @ddt.data('admin', 'user')
    def test_absolute_limits(self, role):
        self.clients[role].manila('absolute-limits')