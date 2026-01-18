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
def _test_validate_bdm_v2(self, stack_name, bdm_v2, with_image=True, error_msg=None, raise_exc=None):
    tmpl, stack = self._setup_test_stack(stack_name)
    if not with_image:
        del tmpl['Resources']['WebServer']['Properties']['image']
    wsp = tmpl.t['Resources']['WebServer']['Properties']
    wsp['block_device_mapping_v2'] = bdm_v2
    resource_defns = tmpl.resource_definitions(stack)
    server = servers.Server('server_create_image_err', resource_defns['WebServer'], stack)
    self.patchobject(nova.NovaClientPlugin, 'get_flavor', return_value=self.mock_flavor)
    self.patchobject(glance.GlanceClientPlugin, 'get_image', return_value=self.mock_image)
    self.stub_VolumeConstraint_validate()
    if raise_exc:
        ex = self.assertRaises(raise_exc, server.validate)
        self.assertIn(error_msg, str(ex))
    else:
        self.assertIsNone(server.validate())