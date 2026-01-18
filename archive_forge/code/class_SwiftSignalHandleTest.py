import datetime
import json
from unittest import mock
import uuid
from oslo_utils import timeutils
from swiftclient import client as swiftclient_client
from swiftclient import exceptions as swiftclient_exceptions
from testtools import matchers
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import swift
from heat.engine import node_data
from heat.engine import resource
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack
from heat.engine import template as templatem
from heat.tests import common
from heat.tests import utils
class SwiftSignalHandleTest(common.HeatTestCase):

    @mock.patch.object(swift.SwiftClientPlugin, '_create')
    @mock.patch.object(resource.Resource, 'physical_resource_name')
    def test_create(self, mock_name, mock_swift):
        st = create_stack(swiftsignalhandle_template)
        handle = st['test_wait_condition_handle']
        mock_swift_object = mock.Mock()
        mock_swift.return_value = mock_swift_object
        mock_swift_object.head_account.return_value = {'x-account-meta-temp-url-key': '1234'}
        mock_swift_object.url = 'http://fake-host.com:8080/v1/AUTH_1234'
        obj_name = '%s-%s-abcdefghijkl' % (st.name, handle.name)
        mock_name.return_value = obj_name
        mock_swift_object.get_container.return_value = cont_index(obj_name, 2)
        mock_swift_object.get_object.return_value = (obj_header, b'{"id": "1"}')
        st.create()
        handle = st.resources['test_wait_condition_handle']
        obj_name = '%s-%s-abcdefghijkl' % (st.name, handle.name)
        regexp = 'http://fake-host.com:8080/v1/AUTH_test_tenant/%s/test_st-test_wait_condition_handle-abcdefghijkl\\?temp_url_sig=[0-9a-f]{40,64}&temp_url_expires=[0-9]{10}' % st.id
        res_id = st.resources['test_wait_condition_handle'].resource_id
        self.assertEqual(res_id, handle.physical_resource_name())
        self.assertThat(handle.FnGetRefId(), matchers.MatchesRegex(regexp))
        self.assertFalse(mock_swift_object.post_account.called)
        header = {'x-versions-location': st.id}
        self.assertEqual({'headers': header}, mock_swift_object.put_container.call_args[1])

    @mock.patch.object(swift.SwiftClientPlugin, '_create')
    @mock.patch.object(resource.Resource, 'physical_resource_name')
    def test_delete_empty_container(self, mock_name, mock_swift):
        st = create_stack(swiftsignalhandle_template)
        handle = st['test_wait_condition_handle']
        mock_swift_object = mock.Mock()
        mock_swift.return_value = mock_swift_object
        mock_swift_object.head_account.return_value = {'x-account-meta-temp-url-key': '1234'}
        mock_swift_object.url = 'http://fake-host.com:8080/v1/AUTH_1234'
        obj_name = '%s-%s-abcdefghijkl' % (st.name, handle.name)
        mock_name.return_value = obj_name
        st.create()
        exc = swiftclient_exceptions.ClientException('Object DELETE failed', http_status=404)
        mock_swift_object.delete_object.side_effect = (None, None, None, exc)
        exc = swiftclient_exceptions.ClientException('Container DELETE failed', http_status=404)
        mock_swift_object.delete_container.side_effect = exc
        rsrc = st.resources['test_wait_condition_handle']
        scheduler.TaskRunner(rsrc.delete)()
        self.assertEqual(('DELETE', 'COMPLETE'), rsrc.state)
        self.assertEqual(4, mock_swift_object.delete_object.call_count)

    @mock.patch.object(swift.SwiftClientPlugin, '_create')
    @mock.patch.object(resource.Resource, 'physical_resource_name')
    def test_delete_object_error(self, mock_name, mock_swift):
        st = create_stack(swiftsignalhandle_template)
        handle = st['test_wait_condition_handle']
        mock_swift_object = mock.Mock()
        mock_swift.return_value = mock_swift_object
        mock_swift_object.head_account.return_value = {'x-account-meta-temp-url-key': '1234'}
        mock_swift_object.url = 'http://fake-host.com:8080/v1/AUTH_1234'
        obj_name = '%s-%s-abcdefghijkl' % (st.name, handle.name)
        mock_name.return_value = obj_name
        st.create()
        exc = swiftclient_exceptions.ClientException('Overlimit', http_status=413)
        mock_swift_object.delete_object.side_effect = (None, None, None, exc)
        rsrc = st.resources['test_wait_condition_handle']
        exc = self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(rsrc.delete))
        self.assertEqual('ClientException: resources.test_wait_condition_handle: Overlimit: 413', str(exc))

    @mock.patch.object(swift.SwiftClientPlugin, '_create')
    @mock.patch.object(resource.Resource, 'physical_resource_name')
    def test_delete_container_error(self, mock_name, mock_swift):
        st = create_stack(swiftsignalhandle_template)
        handle = st['test_wait_condition_handle']
        mock_swift_object = mock.Mock()
        mock_swift.return_value = mock_swift_object
        mock_swift_object.head_account.return_value = {'x-account-meta-temp-url-key': '1234'}
        mock_swift_object.url = 'http://fake-host.com:8080/v1/AUTH_1234'
        obj_name = '%s-%s-abcdefghijkl' % (st.name, handle.name)
        mock_name.return_value = obj_name
        st.create()
        exc = swiftclient_exceptions.ClientException('Object DELETE failed', http_status=404)
        mock_swift_object.delete_object.side_effect = (None, None, None, exc)
        exc = swiftclient_exceptions.ClientException('Overlimit', http_status=413)
        mock_swift_object.delete_container.side_effect = (exc,)
        rsrc = st.resources['test_wait_condition_handle']
        exc = self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(rsrc.delete))
        self.assertEqual('ClientException: resources.test_wait_condition_handle: Overlimit: 413', str(exc))

    @mock.patch.object(swift.SwiftClientPlugin, '_create')
    @mock.patch.object(resource.Resource, 'physical_resource_name')
    def test_delete_non_empty_container(self, mock_name, mock_swift):
        st = create_stack(swiftsignalhandle_template)
        handle = st['test_wait_condition_handle']
        mock_swift_object = mock.Mock()
        mock_swift.return_value = mock_swift_object
        mock_swift_object.head_account.return_value = {'x-account-meta-temp-url-key': '1234'}
        mock_swift_object.url = 'http://fake-host.com:8080/v1/AUTH_1234'
        obj_name = '%s-%s-abcdefghijkl' % (st.name, handle.name)
        mock_name.return_value = obj_name
        st.create()
        exc = swiftclient_exceptions.ClientException('Object DELETE failed', http_status=404)
        mock_swift_object.delete_object.side_effect = (None, None, None, exc)
        exc = swiftclient_exceptions.ClientException('Container DELETE failed', http_status=409)
        mock_swift_object.delete_container.side_effect = exc
        rsrc = st.resources['test_wait_condition_handle']
        scheduler.TaskRunner(rsrc.delete)()
        self.assertEqual(('DELETE', 'COMPLETE'), rsrc.state)
        self.assertEqual(4, mock_swift_object.delete_object.call_count)

    @mock.patch.object(swift.SwiftClientPlugin, '_create')
    def test_handle_update(self, mock_swift):
        st = create_stack(swiftsignalhandle_template)
        handle = st['test_wait_condition_handle']
        mock_swift_object = mock.Mock()
        mock_swift.return_value = mock_swift_object
        mock_swift_object.head_account.return_value = {'x-account-meta-temp-url-key': '1234'}
        mock_swift_object.url = 'http://fake-host.com:8080/v1/AUTH_1234'
        st.create()
        rsrc = st.resources['test_wait_condition_handle']
        old_url = rsrc.FnGetRefId()
        update_snippet = rsrc_defn.ResourceDefinition(handle.name, handle.type(), handle.properties.data)
        scheduler.TaskRunner(handle.update, update_snippet)()
        self.assertEqual(old_url, rsrc.FnGetRefId())

    def test_swift_handle_refid_convergence_cache_data(self):
        cache_data = {'test_wait_condition_handle': node_data.NodeData.from_dict({'uuid': mock.ANY, 'id': mock.ANY, 'action': 'CREATE', 'status': 'COMPLETE', 'reference_id': 'convg_xyz'})}
        st = create_stack(swiftsignalhandle_template, cache_data=cache_data)
        rsrc = st.defn['test_wait_condition_handle']
        self.assertEqual('convg_xyz', rsrc.FnGetRefId())