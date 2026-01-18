import contextlib
import json
from unittest import mock
import uuid
from oslo_config import cfg
from oslo_messaging import exceptions as msg_exceptions
from oslo_serialization import jsonutils
from heat.common import exception
from heat.common import identifier
from heat.common import template_format
from heat.engine import node_data
from heat.engine import resource
from heat.engine.resources import stack_resource
from heat.engine import stack as parser
from heat.engine import template as templatem
from heat.objects import raw_template
from heat.objects import stack as stack_object
from heat.objects import stack_lock
from heat.rpc import api as rpc_api
from heat.tests import common
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
class WithTemplateTest(StackResourceBaseTest):
    scenarios = [('basic', dict(params={}, timeout_mins=None, adopt_data=None)), ('params', dict(params={'foo': 'fee'}, timeout_mins=None, adopt_data=None)), ('timeout', dict(params={}, timeout_mins=53, adopt_data=None)), ('adopt', dict(params={}, timeout_mins=None, adopt_data={'template': 'foo', 'environment': 'eee'}))]

    class IntegerMatch(object):

        def __eq__(self, other):
            if getattr(self, 'match', None) is not None:
                return other == self.match
            if not isinstance(other, int):
                return False
            self.match = other
            return True

        def __ne__(self, other):
            return not self.__eq__(other)

    def test_create_with_template(self):
        child_env = {'parameter_defaults': {}, 'event_sinks': [], 'parameters': self.params, 'resource_registry': {'resources': {}}}
        self.parent_resource.child_params = mock.Mock(return_value=self.params)
        res_name = self.parent_resource.physical_resource_name()
        rpcc = mock.Mock()
        self.parent_resource.rpc_client = rpcc
        rpcc.return_value._create_stack.return_value = {'stack_id': 'pancakes'}
        self.parent_resource.create_with_template(self.empty_temp, user_params=self.params, timeout_mins=self.timeout_mins, adopt_data=self.adopt_data)
        if self.adopt_data is not None:
            adopt_data_str = json.dumps(self.adopt_data)
            tmpl_args = {'template': self.empty_temp.t, 'params': child_env, 'files': {}}
        else:
            adopt_data_str = None
            tmpl_args = {'template_id': self.IntegerMatch(), 'template': None, 'params': None, 'files': None}
        rpcc.return_value._create_stack.assert_called_once_with(self.ctx, stack_name=res_name, args={rpc_api.PARAM_DISABLE_ROLLBACK: True, rpc_api.PARAM_ADOPT_STACK_DATA: adopt_data_str, rpc_api.PARAM_TIMEOUT: self.timeout_mins}, environment_files=None, stack_user_project_id='aprojectid', parent_resource_name='test', user_creds_id='uc123', owner_id=self.parent_stack.id, nested_depth=1, **tmpl_args)

    def test_create_with_template_failure(self):

        class StackValidationFailed_Remote(exception.StackValidationFailed):
            pass
        child_env = {'parameter_defaults': {}, 'event_sinks': [], 'parameters': self.params, 'resource_registry': {'resources': {}}}
        self.parent_resource.child_params = mock.Mock(return_value=self.params)
        res_name = self.parent_resource.physical_resource_name()
        rpcc = mock.Mock()
        self.parent_resource.rpc_client = rpcc
        remote_exc = StackValidationFailed_Remote(message='oops')
        rpcc.return_value._create_stack.side_effect = remote_exc
        self.assertRaises(exception.ResourceFailure, self.parent_resource.create_with_template, self.empty_temp, user_params=self.params, timeout_mins=self.timeout_mins, adopt_data=self.adopt_data)
        if self.adopt_data is not None:
            adopt_data_str = json.dumps(self.adopt_data)
            tmpl_args = {'template': self.empty_temp.t, 'params': child_env, 'files': {}}
        else:
            adopt_data_str = None
            tmpl_args = {'template_id': self.IntegerMatch(), 'template': None, 'params': None, 'files': None}
        rpcc.return_value._create_stack.assert_called_once_with(self.ctx, stack_name=res_name, args={rpc_api.PARAM_DISABLE_ROLLBACK: True, rpc_api.PARAM_ADOPT_STACK_DATA: adopt_data_str, rpc_api.PARAM_TIMEOUT: self.timeout_mins}, environment_files=None, stack_user_project_id='aprojectid', parent_resource_name='test', user_creds_id='uc123', owner_id=self.parent_stack.id, nested_depth=1, **tmpl_args)
        if self.adopt_data is None:
            stored_tmpl_id = tmpl_args['template_id'].match
            self.assertIsNotNone(stored_tmpl_id)
            self.assertRaises(exception.NotFound, raw_template.RawTemplate.get_by_id, self.ctx, stored_tmpl_id)

    def test_update_with_template(self):
        if self.adopt_data is not None:
            return
        ident = identifier.HeatIdentifier(self.ctx.tenant_id, 'fake_name', 'pancakes')
        self.parent_resource.resource_id = ident.stack_id
        self.parent_resource.nested_identifier = mock.Mock(return_value=ident)
        self.parent_resource.child_params = mock.Mock(return_value=self.params)
        rpcc = mock.Mock()
        self.parent_resource.rpc_client = rpcc
        rpcc.return_value._update_stack.return_value = dict(ident)
        status = ('CREATE', 'COMPLETE', '', 'now_time')
        with self.patchobject(stack_object.Stack, 'get_status', return_value=status):
            self.parent_resource.update_with_template(self.empty_temp, user_params=self.params, timeout_mins=self.timeout_mins)
        rpcc.return_value._update_stack.assert_called_once_with(self.ctx, stack_identity=dict(ident), template_id=self.IntegerMatch(), template=None, params=None, files=None, args={rpc_api.PARAM_TIMEOUT: self.timeout_mins, rpc_api.PARAM_CONVERGE: False})

    def test_update_with_template_failure(self):

        class StackValidationFailed_Remote(exception.StackValidationFailed):
            pass
        if self.adopt_data is not None:
            return
        ident = identifier.HeatIdentifier(self.ctx.tenant_id, 'fake_name', 'pancakes')
        self.parent_resource.resource_id = ident.stack_id
        self.parent_resource.nested_identifier = mock.Mock(return_value=ident)
        self.parent_resource.child_params = mock.Mock(return_value=self.params)
        rpcc = mock.Mock()
        self.parent_resource.rpc_client = rpcc
        remote_exc = StackValidationFailed_Remote(message='oops')
        rpcc.return_value._update_stack.side_effect = remote_exc
        status = ('CREATE', 'COMPLETE', '', 'now_time')
        with self.patchobject(stack_object.Stack, 'get_status', return_value=status):
            self.assertRaises(exception.ResourceFailure, self.parent_resource.update_with_template, self.empty_temp, user_params=self.params, timeout_mins=self.timeout_mins)
        template_id = self.IntegerMatch()
        rpcc.return_value._update_stack.assert_called_once_with(self.ctx, stack_identity=dict(ident), template_id=template_id, template=None, params=None, files=None, args={rpc_api.PARAM_TIMEOUT: self.timeout_mins, rpc_api.PARAM_CONVERGE: False})
        self.assertIsNotNone(template_id.match)
        self.assertRaises(exception.NotFound, raw_template.RawTemplate.get_by_id, self.ctx, template_id.match)