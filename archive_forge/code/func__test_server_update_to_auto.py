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
def _test_server_update_to_auto(self, available_multi_nets=None):
    multi_nets = available_multi_nets or []
    return_server = self.fc.servers.list()[1]
    return_server.id = '5678'
    old_networks = [{'port': '95e25541-d26a-478d-8f36-ae1c8f6b74dc'}]
    server = self._create_test_server(return_server, 'networks_update', networks=old_networks)
    update_props = self.server_props.copy()
    update_props['networks'] = [{'allocate_network': 'auto'}]
    update_template = server.t.freeze(properties=update_props)
    self.patchobject(self.fc.servers, 'get', return_value=return_server)
    poor_interfaces = [create_fake_iface(port='95e25541-d26a-478d-8f36-ae1c8f6b74dc', net='aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa', ip='11.12.13.14')]
    self.patchobject(return_server, 'interface_list', return_value=poor_interfaces)
    self.patchobject(server, '_get_available_networks', return_value=multi_nets)
    mock_detach = self.patchobject(return_server, 'interface_detach')
    mock_attach = self.patchobject(return_server, 'interface_attach')
    updater = scheduler.TaskRunner(server.update, update_template)
    if not multi_nets:
        self.patchobject(nova.NovaClientPlugin, 'check_interface_detach', return_value=True)
        self.patchobject(nova.NovaClientPlugin, 'check_interface_attach', return_value=True)
        auto_allocate_net = '9cfe6c74-c105-4906-9a1f-81d9064e9bca'
        self.patchobject(server, '_auto_allocate_network', return_value=[auto_allocate_net])
        updater()
        self.assertEqual((server.UPDATE, server.COMPLETE), server.state)
        self.assertEqual(1, mock_detach.call_count)
        self.assertEqual(1, mock_attach.call_count)
        mock_attach.assert_called_once_with(None, [auto_allocate_net], None)
    else:
        self.assertRaises(exception.ResourceFailure, updater)
        self.assertEqual(0, mock_detach.call_count)
        self.assertEqual(0, mock_attach.call_count)