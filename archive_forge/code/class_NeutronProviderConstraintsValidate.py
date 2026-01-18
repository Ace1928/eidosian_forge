from unittest import mock
from neutronclient.common import exceptions as qe
from heat.common import exception
from heat.engine.clients.os import neutron
from heat.engine.clients.os.neutron import lbaas_constraints as lc
from heat.engine.clients.os.neutron import neutron_constraints as nc
from heat.tests import common
from heat.tests import utils
class NeutronProviderConstraintsValidate(common.HeatTestCase):
    scenarios = [('validate_lbaasv2', dict(constraint_class=lc.LBaasV2ProviderConstraint, service_type='LOADBALANCERV2'))]

    def test_provider_validate(self):
        nc = mock.Mock()
        mock_create = self.patchobject(neutron.NeutronClientPlugin, '_create')
        mock_create.return_value = nc
        providers = {'service_providers': [{'service_type': 'LOADBANALCERV2', 'name': 'haproxy'}, {'service_type': 'LOADBANALCER', 'name': 'haproxy'}]}
        nc.list_service_providers.return_value = providers
        constraint = self.constraint_class()
        ctx = utils.dummy_context()
        self.assertTrue(constraint.validate('haproxy', ctx))
        self.assertFalse(constraint.validate('bar', ctx))