from unittest import mock
from magnumclient.osc.v1 import quotas as osc_quotas
from magnumclient.tests.osc.unit.v1 import fakes as magnum_fakes
class TestQuotas(magnum_fakes.TestMagnumClientOSCV1):

    def setUp(self):
        super(TestQuotas, self).setUp()
        self.quotas_mock = self.app.client_manager.container_infra.quotas