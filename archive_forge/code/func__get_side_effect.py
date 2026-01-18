from unittest import mock
from keystoneauth1 import adapter
from openstack.baremetal.v1 import _common
from openstack.baremetal.v1 import node
from openstack import exceptions
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
def _get_side_effect(_self, session):
    self.node.power_state = 'power off'
    self.assertIs(session, self.session)