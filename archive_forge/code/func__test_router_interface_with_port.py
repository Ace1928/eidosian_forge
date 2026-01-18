import copy
from unittest import mock
from neutronclient.common import exceptions as qe
from neutronclient.v2_0 import client as neutronclient
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.resources.openstack.neutron import router
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def _test_router_interface_with_port(self, resolve_port=True):

    def find_rsrc(resource, name_or_id, cmd_resource=None):
        id_mapping = {'router': 'ae478782-53c0-4434-ab16-49900c88016c', 'port': '9577cafd-8e98-4059-a2e6-8a771b4d318e'}
        return id_mapping.get(resource)
    self.find_rsrc_mock.side_effect = find_rsrc
    self.remove_if_mock.side_effect = [None, qe.NeutronClientException(status_code=404)]
    self.stub_PortConstraint_validate()
    self.stub_RouterConstraint_validate()
    t = template_format.parse(neutron_template)
    stack = utils.parse_stack(t)
    rsrc = self.create_router_interface(t, stack, 'router_interface', properties={'router': 'ae478782-53c0-4434-ab16-49900c88016c', 'port': '9577cafd-8e98-4059-a2e6-8a771b4d318e'})
    if not resolve_port:
        self.assertEqual('9577cafd-8e98-4059-a2e6-8a771b4d318e', rsrc.properties.get(rsrc.PORT))
        self.assertIsNone(rsrc.properties.get(rsrc.PORT_ID))
    scheduler.TaskRunner(rsrc.delete)()
    rsrc.state_set(rsrc.CREATE, rsrc.COMPLETE, 'to delete again')
    scheduler.TaskRunner(rsrc.delete)()