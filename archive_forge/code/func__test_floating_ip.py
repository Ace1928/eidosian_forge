import copy
from unittest import mock
from neutronclient.common import exceptions as qe
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.v2_0 import client as neutronclient
from openstack import exceptions
from oslo_utils import excutils
from heat.common import exception
from heat.common import template_format
from heat.common import timeutils
from heat.engine.clients.os import neutron
from heat.engine.hot import functions as hot_funcs
from heat.engine import node_data
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import stk_defn
from heat.engine import template as tmpl
from heat.tests import common
from heat.tests import utils
def _test_floating_ip(self, tmpl, r_iface=True):
    self.mockclient.create_floatingip.return_value = {'floatingip': {'id': 'fc68ea2c-b60b-4b4f-bd82-94ec81110766', 'floating_network_id': u'abcd1234'}}
    self.mockclient.show_floatingip.side_effect = [qe.NeutronClientException(status_code=404), {'floatingip': {'id': 'fc68ea2c-b60b-4b4f-bd82-94ec81110766', 'floating_network_id': u'abcd1234'}}, {'floatingip': {'id': 'fc68ea2c-b60b-4b4f-bd82-94ec81110766', 'floating_network_id': u'abcd1234'}}, {'floatingip': {'id': 'fc68ea2c-b60b-4b4f-bd82-94ec81110766', 'floating_network_id': u'abcd1234'}}, qe.NeutronClientException(status_code=404)]
    retry_delay = self.patchobject(timeutils, 'retry_backoff_delay', return_value=0.01)
    self.mockclient.delete_floatingip.side_effect = [None, None, qe.NeutronClientException(status_code=404)]
    self.stub_NetworkConstraint_validate()
    stack = utils.parse_stack(tmpl)
    if r_iface:
        required_by = set(stack.dependencies.required_by(stack['router_interface']))
        self.assertIn(stack['floating_ip_assoc'], required_by)
    else:
        deps = stack.dependencies[stack['gateway']]
        self.assertIn(stack['floating_ip'], deps)
    fip = stack['floating_ip']
    scheduler.TaskRunner(fip.create)()
    self.assertEqual((fip.CREATE, fip.COMPLETE), fip.state)
    fip.validate()
    fip_id = fip.FnGetRefId()
    self.assertEqual('fc68ea2c-b60b-4b4f-bd82-94ec81110766', fip_id)
    self.assertIsNone(fip.FnGetAtt('show'))
    self.assertEqual('fc68ea2c-b60b-4b4f-bd82-94ec81110766', fip.FnGetAtt('show')['id'])
    self.assertRaises(exception.InvalidTemplateAttribute, fip.FnGetAtt, 'Foo')
    self.assertEqual(u'abcd1234', fip.FnGetAtt('floating_network_id'))
    scheduler.TaskRunner(fip.delete)()
    fip.state_set(fip.CREATE, fip.COMPLETE, 'to delete again')
    scheduler.TaskRunner(fip.delete)()
    self.mockclient.create_floatingip.assert_called_once_with({'floatingip': {'floating_network_id': u'abcd1234'}})
    self.mockclient.show_floatingip.assert_called_with('fc68ea2c-b60b-4b4f-bd82-94ec81110766')
    retry_delay.assert_called_once_with(1, jitter_max=2.0)
    self.mockclient.delete_floatingip.assert_called_with('fc68ea2c-b60b-4b4f-bd82-94ec81110766')