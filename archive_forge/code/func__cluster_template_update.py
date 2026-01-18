import copy
from unittest import mock
from neutronclient.neutron import v2_0 as neutronV20
from heat.common import exception
from heat.common import template_format
from heat.engine import resource
from heat.engine.resources.openstack.magnum import cluster_template
from heat.engine import scheduler
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def _cluster_template_update(self, update_status='UPDATE_COMPLETE', exc_msg=None):
    ct = self._create_resource('ct', self.rsrc_defn, self.stack)
    status = mock.MagicMock(status=update_status)
    self.client.cluster_templates.get.return_value = status
    t = template_format.parse(self.magnum_template)
    new_t = copy.deepcopy(t)
    new_t['resources'][self.expected['name']]['properties'][cluster_template.ClusterTemplate.PUBLIC] = False
    rsrc_defns = template.Template(new_t).resource_definitions(self.stack)
    new_ct = rsrc_defns[self.expected['name']]
    if update_status == 'UPDATE_COMPLETE':
        scheduler.TaskRunner(ct.update, new_ct)()
        self.assertEqual((ct.UPDATE, ct.COMPLETE), ct.state)
    else:
        exc = self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(ct.update, new_ct))
        self.assertIn(exc_msg, str(exc))