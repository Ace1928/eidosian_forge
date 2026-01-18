from unittest import mock
import uuid
from oslo_config import cfg
from troveclient import exceptions as troveexc
from troveclient.v1 import users
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.clients.os import trove
from heat.engine import resource
from heat.engine.resources.openstack.trove import instance as dbinstance
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import template as tmpl
from heat.tests import common
from heat.tests import utils
def _stubout_validate(self, instance, neutron=None, mock_net_constraint=False, with_port=True):
    if mock_net_constraint:
        self.stub_NetworkConstraint_validate()
    self.client.datastore_versions.list.return_value = [FakeVersion()]
    if neutron is not None:
        instance.is_using_neutron = mock.Mock(return_value=bool(neutron))
        if with_port:
            self.stub_PortConstraint_validate()