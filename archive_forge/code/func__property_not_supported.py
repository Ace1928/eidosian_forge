import copy
from unittest import mock
from ironicclient.common.apiclient import exceptions as ic_exc
from oslo_config import cfg
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import ironic as ic
from heat.engine import resource
from heat.engine.resources.openstack.ironic import port
from heat.engine import scheduler
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def _property_not_supported(self, property_name, version):
    t = template_format.parse(min_port_template)
    new_t = copy.deepcopy(t)
    new_t['resources'][self.fake_name]['properties'][property_name] = self.rsrc_defn._properties[property_name]
    rsrc_defns = template.Template(new_t).resource_definitions(self.stack)
    new_port = rsrc_defns[self.fake_name]
    p = self._create_resource('port-with-%s' % property_name, new_port, self.stack)
    p.client_plugin().max_microversion = version - 0.01
    feature = 'OS::Ironic::Port with %s property' % property_name
    err = self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(p.create))
    self.assertEqual('NotSupported: resources.port-with-%(key)s: %(feature)s is not supported.' % {'feature': feature, 'key': property_name}, str(err))