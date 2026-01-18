import collections
import datetime
import itertools
import json
import os
import sys
from unittest import mock
import uuid
import eventlet
from oslo_config import cfg
from heat.common import exception
from heat.common.i18n import _
from heat.common import short_id
from heat.common import timeutils
from heat.db import api as db_api
from heat.db import models
from heat.engine import attributes
from heat.engine.cfn import functions as cfn_funcs
from heat.engine import clients
from heat.engine import constraints
from heat.engine import dependencies
from heat.engine import environment
from heat.engine import node_data
from heat.engine import plugin_manager
from heat.engine import properties
from heat.engine import resource
from heat.engine import resources
from heat.engine.resources.openstack.heat import none_resource
from heat.engine.resources.openstack.heat import test_resource
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import support
from heat.engine import template
from heat.engine import translation
from heat.objects import resource as resource_objects
from heat.objects import resource_data as resource_data_object
from heat.objects import resource_properties_data as rpd_object
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
import neutronclient.common.exceptions as neutron_exp
class ResourceAvailabilityTest(common.HeatTestCase):

    def _mock_client_plugin(self, service_types=None, is_available=True):
        service_types = service_types or []
        mock_client_plugin = mock.Mock()
        mock_service_types = mock.PropertyMock(return_value=service_types)
        type(mock_client_plugin).service_types = mock_service_types
        mock_client_plugin.does_endpoint_exist = mock.Mock(return_value=is_available)
        return (mock_service_types, mock_client_plugin)

    def test_default_true_with_default_client_name_none(self):
        """Test availability of resource when default_client_name is None.

        When default_client_name is None, resource is considered as available.
        """
        with mock.patch('heat.tests.generic_resource.ResourceWithDefaultClientName.default_client_name', new_callable=mock.PropertyMock) as mock_client_name:
            mock_client_name.return_value = None
            self.assertTrue(generic_rsrc.ResourceWithDefaultClientName.is_service_available(context=mock.Mock())[0])

    @mock.patch.object(clients.OpenStackClients, 'client_plugin')
    def test_default_true_empty_service_types(self, mock_client_plugin_method):
        """Test availability of resource when service_types is empty list.

        When service_types is empty list, resource is considered as available.
        """
        mock_service_types, mock_client_plugin = self._mock_client_plugin()
        mock_client_plugin_method.return_value = mock_client_plugin
        self.assertTrue(generic_rsrc.ResourceWithDefaultClientName.is_service_available(context=mock.Mock())[0])
        mock_client_plugin_method.assert_called_once_with(generic_rsrc.ResourceWithDefaultClientName.default_client_name)
        mock_service_types.assert_called_once_with()

    @mock.patch.object(clients.OpenStackClients, 'client_plugin')
    def test_service_deployed(self, mock_client_plugin_method):
        """Test availability of resource when the service is deployed.

        When the service is deployed, resource is considered as available.
        """
        mock_service_types, mock_client_plugin = self._mock_client_plugin(['test_type'])
        mock_client_plugin_method.return_value = mock_client_plugin
        self.assertTrue(generic_rsrc.ResourceWithDefaultClientName.is_service_available(context=mock.Mock())[0])
        mock_client_plugin_method.assert_called_once_with(generic_rsrc.ResourceWithDefaultClientName.default_client_name)
        mock_service_types.assert_called_once_with()
        mock_client_plugin.does_endpoint_exist.assert_called_once_with(service_type='test_type', service_name=generic_rsrc.ResourceWithDefaultClientName.default_client_name)

    @mock.patch.object(clients.OpenStackClients, 'client_plugin')
    def test_service_not_deployed(self, mock_client_plugin_method):
        """Test availability of resource when the service is not deployed.

        When the service is not deployed, resource is considered as
        unavailable.
        """
        mock_service_types, mock_client_plugin = self._mock_client_plugin(['test_type_un_deployed'], False)
        mock_client_plugin_method.return_value = mock_client_plugin
        self.assertFalse(generic_rsrc.ResourceWithDefaultClientName.is_service_available(context=mock.Mock())[0])
        mock_client_plugin_method.assert_called_once_with(generic_rsrc.ResourceWithDefaultClientName.default_client_name)
        mock_service_types.assert_called_once_with()
        mock_client_plugin.does_endpoint_exist.assert_called_once_with(service_type='test_type_un_deployed', service_name=generic_rsrc.ResourceWithDefaultClientName.default_client_name)

    @mock.patch.object(clients.OpenStackClients, 'client_plugin')
    def test_service_deployed_required_extension_true(self, mock_client_plugin_method):
        """Test availability of resource with a required extension. """
        mock_service_types, mock_client_plugin = self._mock_client_plugin(['test_type'])
        mock_client_plugin.has_extension = mock.Mock(return_value=True)
        mock_client_plugin_method.return_value = mock_client_plugin
        self.assertTrue(generic_rsrc.ResourceWithDefaultClientNameExt.is_service_available(context=mock.Mock())[0])
        mock_client_plugin_method.assert_called_once_with(generic_rsrc.ResourceWithDefaultClientName.default_client_name)
        mock_service_types.assert_called_once_with()
        mock_client_plugin.does_endpoint_exist.assert_called_once_with(service_type='test_type', service_name=generic_rsrc.ResourceWithDefaultClientName.default_client_name)
        mock_client_plugin.has_extension.assert_called_once_with('foo')

    @mock.patch.object(clients.OpenStackClients, 'client_plugin')
    def test_service_deployed_required_extension_false(self, mock_client_plugin_method):
        """Test availability of resource with a required extension. """
        mock_service_types, mock_client_plugin = self._mock_client_plugin(['test_type'])
        mock_client_plugin.has_extension = mock.Mock(return_value=False)
        mock_client_plugin_method.return_value = mock_client_plugin
        self.assertFalse(generic_rsrc.ResourceWithDefaultClientNameExt.is_service_available(context=mock.Mock())[0])
        mock_client_plugin_method.assert_called_once_with(generic_rsrc.ResourceWithDefaultClientName.default_client_name)
        mock_service_types.assert_called_once_with()
        mock_client_plugin.does_endpoint_exist.assert_called_once_with(service_type='test_type', service_name=generic_rsrc.ResourceWithDefaultClientName.default_client_name)
        mock_client_plugin.has_extension.assert_called_once_with('foo')

    @mock.patch.object(clients.OpenStackClients, 'client_plugin')
    def test_service_deployed_required_extension_exception(self, mock_client_plugin_method):
        """Test availability of resource with a required extension. """
        mock_service_types, mock_client_plugin = self._mock_client_plugin(['test_type'])
        mock_client_plugin.has_extension = mock.Mock(side_effect=exception.AuthorizationFailure())
        mock_client_plugin_method.return_value = mock_client_plugin
        self.assertRaises(exception.AuthorizationFailure, generic_rsrc.ResourceWithDefaultClientNameExt.is_service_available, context=mock.Mock())
        mock_client_plugin_method.assert_called_once_with(generic_rsrc.ResourceWithDefaultClientName.default_client_name)
        mock_service_types.assert_called_once_with()
        mock_client_plugin.does_endpoint_exist.assert_called_once_with(service_type='test_type', service_name=generic_rsrc.ResourceWithDefaultClientName.default_client_name)
        mock_client_plugin.has_extension.assert_called_once_with('foo')

    @mock.patch.object(clients.OpenStackClients, 'client_plugin')
    def test_service_not_deployed_required_extension(self, mock_client_plugin_method):
        """Test availability of resource when the service is not deployed.

        When the service is not deployed, resource is considered as
        unavailable.
        """
        mock_service_types, mock_client_plugin = self._mock_client_plugin(['test_type_un_deployed'], False)
        mock_client_plugin_method.return_value = mock_client_plugin
        self.assertFalse(generic_rsrc.ResourceWithDefaultClientNameExt.is_service_available(context=mock.Mock())[0])
        mock_client_plugin_method.assert_called_once_with(generic_rsrc.ResourceWithDefaultClientName.default_client_name)
        mock_service_types.assert_called_once_with()
        mock_client_plugin.does_endpoint_exist.assert_called_once_with(service_type='test_type_un_deployed', service_name=generic_rsrc.ResourceWithDefaultClientName.default_client_name)

    @mock.patch.object(clients.OpenStackClients, 'client_plugin')
    def test_service_not_installed_required_extension(self, mock_client_plugin_method):
        """Test availability of resource when the client is not installed.

        When the client is not installed, we can't create the resource properly
        so raise a proper exception for information.
        """
        mock_client_plugin_method.return_value = None
        self.assertRaises(exception.ClientNotAvailable, generic_rsrc.ResourceWithDefaultClientNameExt.is_service_available, context=mock.Mock())
        mock_client_plugin_method.assert_called_once_with(generic_rsrc.ResourceWithDefaultClientName.default_client_name)

    def test_service_not_available_returns_false(self):
        """Test when the service is not in service catalog.

        When the service is not deployed, make sure resource is throwing
        ResourceTypeUnavailable exception.
        """
        with mock.patch.object(generic_rsrc.ResourceWithDefaultClientName, 'is_service_available') as mock_method:
            mock_method.return_value = (False, 'Service endpoint not in service catalog.')
            definition = rsrc_defn.ResourceDefinition(name='Test Resource', resource_type='UnavailableResourceType')
            mock_stack = mock.MagicMock()
            mock_stack.in_convergence_check = False
            mock_stack.db_resource_get.return_value = None
            rsrc = generic_rsrc.ResourceWithDefaultClientName('test_res', definition, mock_stack)
            ex = self.assertRaises(exception.ResourceTypeUnavailable, rsrc.validate_template)
            msg = 'HEAT-E99001 Service sample is not available for resource type UnavailableResourceType, reason: Service endpoint not in service catalog.'
            self.assertEqual(msg, str(ex), 'invalid exception message')
            mock_method.assert_called_once_with(mock_stack.context)

    def test_service_not_available_throws_exception(self):
        """Test for other exceptions when checking for service availability

        Ex. when client throws an error, make sure resource is throwing
        ResourceTypeUnavailable that contains the original exception message.
        """
        with mock.patch.object(generic_rsrc.ResourceWithDefaultClientName, 'is_service_available') as mock_method:
            mock_method.side_effect = exception.AuthorizationFailure()
            definition = rsrc_defn.ResourceDefinition(name='Test Resource', resource_type='UnavailableResourceType')
            mock_stack = mock.MagicMock()
            mock_stack.in_convergence_check = False
            mock_stack.db_resource_get.return_value = None
            rsrc = generic_rsrc.ResourceWithDefaultClientName('test_res', definition, mock_stack)
            ex = self.assertRaises(exception.ResourceTypeUnavailable, rsrc.validate_template)
            msg = 'HEAT-E99001 Service sample is not available for resource type UnavailableResourceType, reason: Authorization failed.'
            self.assertEqual(msg, str(ex), 'invalid exception message')
            mock_method.assert_called_once_with(mock_stack.context)

    def test_handle_delete_successful(self):
        self.stack = parser.Stack(utils.dummy_context(), 'test_stack', template.Template(empty_template))
        self.stack.store()
        snippet = rsrc_defn.ResourceDefinition('aresource', 'OS::Heat::None')
        res = resource.Resource('aresource', snippet, self.stack)
        FakeClient = collections.namedtuple('Client', ['entity'])
        client = FakeClient(collections.namedtuple('entity', ['delete']))
        self.patchobject(resource.Resource, 'client', return_value=client)
        delete = mock.Mock()
        res.client().entity.delete = delete
        res.entity = 'entity'
        res.default_client_name = 'something'
        res.resource_id = '12345'
        self.assertEqual('12345', res.handle_delete())
        delete.assert_called_once_with('12345')

    def test_handle_delete_not_found(self):
        self.stack = parser.Stack(utils.dummy_context(), 'test_stack', template.Template(empty_template))
        self.stack.store()
        snippet = rsrc_defn.ResourceDefinition('aresource', 'OS::Heat::None')
        res = resource.Resource('aresource', snippet, self.stack)
        res.entity = 'entity'
        res.default_client_name = 'foo'
        res.resource_id = '12345'
        FakeClient = collections.namedtuple('Client', ['entity'])
        client = FakeClient(collections.namedtuple('entity', ['delete']))
        client_plugin = res._default_client_plugin()

        def is_not_found(ex):
            return isinstance(ex, exception.NotFound)
        client_plugin.is_not_found = mock.Mock(side_effect=is_not_found)
        self.patchobject(resource.Resource, 'client', return_value=client)
        delete = mock.Mock()
        delete.side_effect = [exception.NotFound()]
        res.client().entity.delete = delete
        with mock.patch.object(res, '_default_client_plugin', return_value=client_plugin):
            self.assertIsNone(res.handle_delete())
        delete.assert_called_once_with('12345')

    def test_handle_delete_raise_error(self):
        self.stack = parser.Stack(utils.dummy_context(), 'test_stack', template.Template(empty_template))
        self.stack.store()
        snippet = rsrc_defn.ResourceDefinition('aresource', 'OS::Heat::None')
        res = resource.Resource('aresource', snippet, self.stack)
        res.entity = 'entity'
        res.default_client_name = 'something'
        res.resource_id = '12345'
        FakeClient = collections.namedtuple('Client', ['entity'])
        client = FakeClient(collections.namedtuple('entity', ['delete']))
        client_plugin = res._default_client_plugin()

        def is_not_found(ex):
            return isinstance(ex, exception.NotFound)
        client_plugin.is_not_found = mock.Mock(side_effect=is_not_found)
        self.patchobject(resource.Resource, 'client', return_value=client)
        delete = mock.Mock()
        delete.side_effect = [exception.Error('boom!')]
        res.client().entity.delete = delete
        with mock.patch.object(res, '_default_client_plugin', return_value=client_plugin):
            ex = self.assertRaises(exception.Error, res.handle_delete)
        self.assertEqual('boom!', str(ex))
        delete.assert_called_once_with('12345')

    def test_handle_delete_no_entity(self):
        self.stack = parser.Stack(utils.dummy_context(), 'test_stack', template.Template(empty_template))
        self.stack.store()
        snippet = rsrc_defn.ResourceDefinition('aresource', 'OS::Heat::None')
        res = resource.Resource('aresource', snippet, self.stack)
        FakeClient = collections.namedtuple('Client', ['entity'])
        client = FakeClient(collections.namedtuple('entity', ['delete']))
        self.patchobject(resource.Resource, 'client', return_value=client)
        delete = mock.Mock()
        res.client().entity.delete = delete
        res.resource_id = '12345'
        self.assertIsNone(res.handle_delete())
        self.assertEqual(0, delete.call_count)

    def test_handle_delete_no_resource_id(self):
        self.stack = parser.Stack(utils.dummy_context(), 'test_stack', template.Template(empty_template))
        self.stack.store()
        snippet = rsrc_defn.ResourceDefinition('aresource', 'OS::Heat::None')
        res = resource.Resource('aresource', snippet, self.stack)
        FakeClient = collections.namedtuple('Client', ['entity'])
        client = FakeClient(collections.namedtuple('entity', ['delete']))
        self.patchobject(resource.Resource, 'client', return_value=client)
        delete = mock.Mock()
        res.client().entity.delete = delete
        res.entity = 'entity'
        res.default_client_name = 'something'
        res.resource_id = None
        self.assertIsNone(res.handle_delete())
        self.assertEqual(0, delete.call_count)

    def test_handle_delete_no_default_client_name(self):

        class MyException(Exception):
            pass
        self.stack = parser.Stack(utils.dummy_context(), 'test_stack', template.Template(empty_template))
        self.stack.store()
        snippet = rsrc_defn.ResourceDefinition('aresource', 'OS::Heat::None')
        res = resource.Resource('aresource', snippet, self.stack)
        FakeClient = collections.namedtuple('Client', ['entity'])
        client = FakeClient(collections.namedtuple('entity', ['delete']))
        self.patchobject(resource.Resource, 'client', return_value=client)
        delete = mock.Mock()
        delete.side_effect = [MyException]
        res.client().entity.delete = delete
        res.entity = 'entity'
        res.resource_id = '1234'
        res.default_client_name = None
        self.assertRaises(MyException, res.handle_delete)

    @mock.patch.object(clients.OpenStackClients, 'client_plugin')
    def test_service_deployed_required_extension_true_string(self, mock_client_plugin_method):
        """Test availability of resource with a required extension. """
        mock_service_types, mock_client_plugin = self._mock_client_plugin(['test_type'])
        mock_client_plugin.has_extension = mock.Mock(side_effect=[True, True])
        mock_client_plugin_method.return_value = mock_client_plugin
        rsrc = generic_rsrc.ResourceWithDefaultClientNameMultiStrExt
        rsrc.is_service_available(context=mock.Mock())[0]
        mock_client_plugin_method.assert_called_once_with(generic_rsrc.ResourceWithDefaultClientName.default_client_name)
        mock_service_types.assert_called_once_with()
        mock_client_plugin.does_endpoint_exist.assert_called_once_with(service_type='test_type', service_name=generic_rsrc.ResourceWithDefaultClientName.default_client_name)
        mock_client_plugin.has_extension.assert_has_calls([mock.call('foo'), mock.call('bar')])

    @mock.patch.object(clients.OpenStackClients, 'client_plugin')
    def test_service_deployed_required_extension_true_list(self, mock_client_plugin_method):
        """Test availability of resource with a required extension. """
        mock_service_types, mock_client_plugin = self._mock_client_plugin(['test_type'])
        mock_client_plugin.has_extension = mock.Mock(side_effect=[True, True])
        mock_client_plugin_method.return_value = mock_client_plugin
        rsrc = generic_rsrc.ResourceWithDefaultClientNameMultiExt
        rsrc.is_service_available(context=mock.Mock())[0]
        mock_client_plugin_method.assert_called_once_with(generic_rsrc.ResourceWithDefaultClientName.default_client_name)
        mock_service_types.assert_called_once_with()
        mock_client_plugin.does_endpoint_exist.assert_called_once_with(service_type='test_type', service_name=generic_rsrc.ResourceWithDefaultClientName.default_client_name)
        mock_client_plugin.has_extension.assert_has_calls([mock.call('foo'), mock.call('bar')])

    @mock.patch.object(clients.OpenStackClients, 'client_plugin')
    def test_service_deployed_required_extension_true_list_fail(self, mock_client_plugin_method):
        """Test availability of resource with a required extension. """
        mock_service_types, mock_client_plugin = self._mock_client_plugin(['test_type'])
        mock_client_plugin.has_extension = mock.Mock(side_effect=[True, False])
        mock_client_plugin_method.return_value = mock_client_plugin
        rsrc = generic_rsrc.ResourceWithDefaultClientNameMultiExt
        self.assertFalse(rsrc.is_service_available(context=mock.Mock())[0])
        mock_client_plugin_method.assert_called_once_with(generic_rsrc.ResourceWithDefaultClientName.default_client_name)
        mock_service_types.assert_called_once_with()
        mock_client_plugin.does_endpoint_exist.assert_called_once_with(service_type='test_type', service_name=generic_rsrc.ResourceWithDefaultClientName.default_client_name)
        mock_client_plugin.has_extension.assert_has_calls([mock.call('foo'), mock.call('bar')])