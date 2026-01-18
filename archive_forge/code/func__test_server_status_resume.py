import collections
import contextlib
import copy
from unittest import mock
from keystoneauth1 import exceptions as ks_exceptions
from neutronclient.v2_0 import client as neutronclient
from novaclient import exceptions as nova_exceptions
from oslo_serialization import jsonutils
from oslo_utils import uuidutils
import requests
from urllib import parse as urlparse
from heat.common import exception
from heat.common.i18n import _
from heat.common import template_format
from heat.engine.clients.os import glance
from heat.engine.clients.os import heat_plugin
from heat.engine.clients.os import neutron
from heat.engine.clients.os import nova
from heat.engine.clients.os import swift
from heat.engine.clients.os import zaqar
from heat.engine import environment
from heat.engine import resource
from heat.engine.resources.openstack.nova import server as servers
from heat.engine.resources.openstack.nova import server_network_mixin
from heat.engine.resources import scheduler_hints as sh
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import template
from heat.objects import resource_data as resource_data_object
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def _test_server_status_resume(self, name, state=('SUSPEND', 'COMPLETE')):
    return_server = self.fc.servers.list()[1]
    server = self._create_test_server(return_server, name)
    server.resource_id = '1234'
    server.state_set(state[0], state[1])
    self.patchobject(return_server, 'resume')
    self.patchobject(self.fc.servers, 'get', side_effect=ServerStatus(return_server, ['SUSPENDED', 'SUSPENDED', 'ACTIVE']))
    scheduler.TaskRunner(server.resume)()
    self.assertEqual((server.RESUME, server.COMPLETE), server.state)