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
def _setup_test_server(self, return_server, name, image_id=None, override_name=False, stub_create=True, networks=None):
    stack_name = '%s_s' % name

    def _mock_find_id(resource, name_or_id, cmd_resource=None):
        return name_or_id
    mock_find = self.patchobject(neutron.NeutronClientPlugin, 'find_resourceid_by_name_or_id')
    mock_find.side_effect = _mock_find_id
    server_name = str(name) if override_name else None
    tmpl, self.stack = self._get_test_template(stack_name, server_name, image_id)
    props = tmpl.t['Resources']['WebServer']['Properties']
    if networks is not None:
        props['networks'] = networks
    self.server_props = props
    resource_defns = tmpl.resource_definitions(self.stack)
    server = servers.Server(str(name), resource_defns['WebServer'], self.stack)
    self.patchobject(server, 'store_external_ports')
    self.patchobject(nova.NovaClientPlugin, 'client', return_value=self.fc)
    self.patchobject(glance.GlanceClientPlugin, 'get_image', return_value=self.mock_image)
    if stub_create:
        self.patchobject(self.fc.servers, 'create', return_value=return_server)
        self.patchobject(self.fc.servers, 'get', return_value=return_server)
    return server