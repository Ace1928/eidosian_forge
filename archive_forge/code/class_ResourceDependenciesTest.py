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
class ResourceDependenciesTest(common.HeatTestCase):

    def setUp(self):
        super(ResourceDependenciesTest, self).setUp()
        self.deps = dependencies.Dependencies()

    def test_no_deps(self):
        tmpl = template.Template({'HeatTemplateFormatVersion': '2012-12-12', 'Resources': {'foo': {'Type': 'GenericResourceType'}}})
        stack = parser.Stack(utils.dummy_context(), 'test', tmpl)
        res = stack['foo']
        res.add_explicit_dependencies(self.deps)
        graph = self.deps.graph()
        self.assertIn(res, graph)

    def test_hot_add_dep_error_create(self):
        tmpl = template.Template({'heat_template_version': '2013-05-23', 'resources': {'foo': {'type': 'GenericResourceType'}, 'bar': {'type': 'ResourceWithPropsType'}}})
        stack = parser.Stack(utils.dummy_context(), 'test', tmpl)
        res = stack['bar']

        class TestException(Exception):
            pass
        self.patchobject(res, 'add_dependencies', side_effect=TestException)

        def get_dependencies():
            return stack.dependencies
        self.assertRaises(TestException, get_dependencies)

    def test_hot_add_dep_error_load(self):
        tmpl = template.Template({'heat_template_version': '2013-05-23', 'resources': {'foo': {'type': 'GenericResourceType'}, 'bar': {'type': 'ResourceWithPropsType'}}})
        stack = parser.Stack(utils.dummy_context(), 'test_hot_add_dep_err', tmpl)
        stack.store()
        res = stack['bar']
        self.patchobject(res, 'add_dependencies', side_effect=ValueError)
        graph = stack.dependencies.graph()
        self.assertIn(res, graph)
        self.assertIn(stack['foo'], graph)

    def test_ref(self):
        tmpl = template.Template({'HeatTemplateFormatVersion': '2012-12-12', 'Resources': {'foo': {'Type': 'GenericResourceType'}, 'bar': {'Type': 'ResourceWithPropsType', 'Properties': {'Foo': {'Ref': 'foo'}}}}})
        stack = parser.Stack(utils.dummy_context(), 'test', tmpl)
        res = stack['bar']
        res.add_explicit_dependencies(self.deps)
        graph = self.deps.graph()
        self.assertIn(res, graph)
        self.assertIn(stack['foo'], graph[res])

    def test_hot_ref(self):
        """Test that HOT get_resource creates dependencies."""
        tmpl = template.Template({'heat_template_version': '2013-05-23', 'resources': {'foo': {'type': 'GenericResourceType'}, 'bar': {'type': 'ResourceWithPropsType', 'properties': {'Foo': {'get_resource': 'foo'}}}}})
        stack = parser.Stack(utils.dummy_context(), 'test', tmpl)
        res = stack['bar']
        res.add_explicit_dependencies(self.deps)
        graph = self.deps.graph()
        self.assertIn(res, graph)
        self.assertIn(stack['foo'], graph[res])

    def test_ref_nested_dict(self):
        tmpl = template.Template({'HeatTemplateFormatVersion': '2012-12-12', 'Resources': {'foo': {'Type': 'GenericResourceType'}, 'bar': {'Type': 'ResourceWithPropsType', 'Properties': {'Foo': {'Fn::Base64': {'Ref': 'foo'}}}}}})
        stack = parser.Stack(utils.dummy_context(), 'test', tmpl)
        res = stack['bar']
        res.add_explicit_dependencies(self.deps)
        graph = self.deps.graph()
        self.assertIn(res, graph)
        self.assertIn(stack['foo'], graph[res])

    def test_hot_ref_nested_dict(self):
        tmpl = template.Template({'heat_template_version': '2013-05-23', 'resources': {'foo': {'type': 'GenericResourceType'}, 'bar': {'type': 'ResourceWithPropsType', 'properties': {'Foo': {'Fn::Base64': {'get_resource': 'foo'}}}}}})
        stack = parser.Stack(utils.dummy_context(), 'test', tmpl)
        res = stack['bar']
        res.add_explicit_dependencies(self.deps)
        graph = self.deps.graph()
        self.assertIn(res, graph)
        self.assertIn(stack['foo'], graph[res])

    def test_ref_nested_deep(self):
        tmpl = template.Template({'HeatTemplateFormatVersion': '2012-12-12', 'Resources': {'foo': {'Type': 'GenericResourceType'}, 'bar': {'Type': 'ResourceWithPropsType', 'Properties': {'Foo': {'Fn::Join': [',', ['blarg', {'Ref': 'foo'}, 'wibble']]}}}}})
        stack = parser.Stack(utils.dummy_context(), 'test', tmpl)
        res = stack['bar']
        res.add_explicit_dependencies(self.deps)
        graph = self.deps.graph()
        self.assertIn(res, graph)
        self.assertIn(stack['foo'], graph[res])

    def test_hot_ref_nested_deep(self):
        tmpl = template.Template({'heat_template_version': '2013-05-23', 'resources': {'foo': {'type': 'GenericResourceType'}, 'bar': {'type': 'ResourceWithPropsType', 'properties': {'foo': {'Fn::Join': [',', ['blarg', {'get_resource': 'foo'}, 'wibble']]}}}}})
        stack = parser.Stack(utils.dummy_context(), 'test', tmpl)
        res = stack['bar']
        res.add_explicit_dependencies(self.deps)
        graph = self.deps.graph()
        self.assertIn(res, graph)
        self.assertIn(stack['foo'], graph[res])

    def test_ref_fail(self):
        tmpl = template.Template({'HeatTemplateFormatVersion': '2012-12-12', 'Resources': {'foo': {'Type': 'GenericResourceType'}, 'bar': {'Type': 'ResourceWithPropsType', 'Properties': {'Foo': {'Ref': 'baz'}}}}})
        stack = parser.Stack(utils.dummy_context(), 'test', tmpl)
        self.assertRaises(exception.StackValidationFailed, stack.validate)

    def test_hot_ref_fail(self):
        tmpl = template.Template({'heat_template_version': '2013-05-23', 'resources': {'foo': {'type': 'GenericResourceType'}, 'bar': {'type': 'ResourceWithPropsType', 'properties': {'Foo': {'get_resource': 'baz'}}}}})
        stack = parser.Stack(utils.dummy_context(), 'test', tmpl)
        ex = self.assertRaises(exception.InvalidTemplateReference, stack.validate)
        self.assertIn('"baz" (in bar.Properties.Foo)', str(ex))

    def test_validate_value_fail(self):
        tmpl = template.Template({'heat_template_version': '2013-05-23', 'resources': {'bar': {'type': 'ResourceWithPropsType', 'properties': {'FooInt': 'notanint'}}}})
        stack = parser.Stack(utils.dummy_context(), 'test', tmpl)
        ex = self.assertRaises(exception.StackValidationFailed, stack.validate)
        self.assertIn("Property error: resources.bar.properties.FooInt: Value 'notanint' is not an integer", str(ex))
        stack_novalidate = parser.Stack(utils.dummy_context(), 'test', tmpl, strict_validate=False)
        self.assertIsNone(stack_novalidate.validate())

    def test_getatt(self):
        tmpl = template.Template({'HeatTemplateFormatVersion': '2012-12-12', 'Resources': {'foo': {'Type': 'GenericResourceType'}, 'bar': {'Type': 'ResourceWithPropsType', 'Properties': {'Foo': {'Fn::GetAtt': ['foo', 'bar']}}}}})
        stack = parser.Stack(utils.dummy_context(), 'test', tmpl)
        res = stack['bar']
        res.add_explicit_dependencies(self.deps)
        graph = self.deps.graph()
        self.assertIn(res, graph)
        self.assertIn(stack['foo'], graph[res])

    def test_hot_getatt(self):
        tmpl = template.Template({'heat_template_version': '2013-05-23', 'resources': {'foo': {'type': 'GenericResourceType'}, 'bar': {'type': 'ResourceWithPropsType', 'properties': {'Foo': {'get_attr': ['foo', 'bar']}}}}})
        stack = parser.Stack(utils.dummy_context(), 'test', tmpl)
        res = stack['bar']
        res.add_explicit_dependencies(self.deps)
        graph = self.deps.graph()
        self.assertIn(res, graph)
        self.assertIn(stack['foo'], graph[res])

    def test_getatt_nested_dict(self):
        tmpl = template.Template({'HeatTemplateFormatVersion': '2012-12-12', 'Resources': {'foo': {'Type': 'GenericResourceType'}, 'bar': {'Type': 'ResourceWithPropsType', 'Properties': {'Foo': {'Fn::Base64': {'Fn::GetAtt': ['foo', 'bar']}}}}}})
        stack = parser.Stack(utils.dummy_context(), 'test', tmpl)
        res = stack['bar']
        res.add_explicit_dependencies(self.deps)
        graph = self.deps.graph()
        self.assertIn(res, graph)
        self.assertIn(stack['foo'], graph[res])

    def test_hot_getatt_nested_dict(self):
        tmpl = template.Template({'heat_template_version': '2013-05-23', 'resources': {'foo': {'type': 'GenericResourceType'}, 'bar': {'type': 'ResourceWithPropsType', 'properties': {'Foo': {'Fn::Base64': {'get_attr': ['foo', 'bar']}}}}}})
        stack = parser.Stack(utils.dummy_context(), 'test', tmpl)
        res = stack['bar']
        res.add_explicit_dependencies(self.deps)
        graph = self.deps.graph()
        self.assertIn(res, graph)
        self.assertIn(stack['foo'], graph[res])

    def test_getatt_nested_deep(self):
        tmpl = template.Template({'HeatTemplateFormatVersion': '2012-12-12', 'Resources': {'foo': {'Type': 'GenericResourceType'}, 'bar': {'Type': 'ResourceWithPropsType', 'Properties': {'Foo': {'Fn::Join': [',', ['blarg', {'Fn::GetAtt': ['foo', 'bar']}, 'wibble']]}}}}})
        stack = parser.Stack(utils.dummy_context(), 'test', tmpl)
        res = stack['bar']
        res.add_explicit_dependencies(self.deps)
        graph = self.deps.graph()
        self.assertIn(res, graph)
        self.assertIn(stack['foo'], graph[res])

    def test_hot_getatt_nested_deep(self):
        tmpl = template.Template({'heat_template_version': '2013-05-23', 'resources': {'foo': {'type': 'GenericResourceType'}, 'bar': {'type': 'ResourceWithPropsType', 'properties': {'Foo': {'Fn::Join': [',', ['blarg', {'get_attr': ['foo', 'bar']}, 'wibble']]}}}}})
        stack = parser.Stack(utils.dummy_context(), 'test', tmpl)
        res = stack['bar']
        res.add_explicit_dependencies(self.deps)
        graph = self.deps.graph()
        self.assertIn(res, graph)
        self.assertIn(stack['foo'], graph[res])

    def test_getatt_fail(self):
        tmpl = template.Template({'HeatTemplateFormatVersion': '2012-12-12', 'Resources': {'foo': {'Type': 'GenericResourceType'}, 'bar': {'Type': 'ResourceWithPropsType', 'Properties': {'Foo': {'Fn::GetAtt': ['baz', 'bar']}}}}})
        stack = parser.Stack(utils.dummy_context(), 'test', tmpl)
        ex = self.assertRaises(exception.InvalidTemplateReference, getattr, stack, 'dependencies')
        self.assertIn('"baz" (in bar.Properties.Foo)', str(ex))

    def test_hot_getatt_fail(self):
        tmpl = template.Template({'heat_template_version': '2013-05-23', 'resources': {'foo': {'type': 'GenericResourceType'}, 'bar': {'type': 'ResourceWithPropsType', 'properties': {'Foo': {'get_attr': ['baz', 'bar']}}}}})
        stack = parser.Stack(utils.dummy_context(), 'test', tmpl)
        ex = self.assertRaises(exception.InvalidTemplateReference, getattr, stack, 'dependencies')
        self.assertIn('"baz" (in bar.Properties.Foo)', str(ex))

    def test_getatt_fail_nested_deep(self):
        tmpl = template.Template({'HeatTemplateFormatVersion': '2012-12-12', 'Resources': {'foo': {'Type': 'GenericResourceType'}, 'bar': {'Type': 'ResourceWithPropsType', 'Properties': {'Foo': {'Fn::Join': [',', ['blarg', {'Fn::GetAtt': ['foo', 'bar']}, 'wibble', {'Fn::GetAtt': ['baz', 'bar']}]]}}}}})
        stack = parser.Stack(utils.dummy_context(), 'test', tmpl)
        ex = self.assertRaises(exception.InvalidTemplateReference, getattr, stack, 'dependencies')
        self.assertIn('"baz" (in bar.Properties.Foo.Fn::Join[1][3])', str(ex))

    def test_hot_getatt_fail_nested_deep(self):
        tmpl = template.Template({'heat_template_version': '2013-05-23', 'resources': {'foo': {'type': 'GenericResourceType'}, 'bar': {'type': 'ResourceWithPropsType', 'properties': {'Foo': {'Fn::Join': [',', ['blarg', {'get_attr': ['foo', 'bar']}, 'wibble', {'get_attr': ['baz', 'bar']}]]}}}}})
        stack = parser.Stack(utils.dummy_context(), 'test', tmpl)
        ex = self.assertRaises(exception.InvalidTemplateReference, getattr, stack, 'dependencies')
        self.assertIn('"baz" (in bar.Properties.Foo.Fn::Join[1][3])', str(ex))

    def test_dependson(self):
        tmpl = template.Template({'HeatTemplateFormatVersion': '2012-12-12', 'Resources': {'foo': {'Type': 'GenericResourceType'}, 'bar': {'Type': 'GenericResourceType', 'DependsOn': 'foo'}}})
        stack = parser.Stack(utils.dummy_context(), 'test', tmpl)
        res = stack['bar']
        res.add_explicit_dependencies(self.deps)
        graph = self.deps.graph()
        self.assertIn(res, graph)
        self.assertIn(stack['foo'], graph[res])

    def test_dependson_hot(self):
        tmpl = template.Template({'heat_template_version': '2013-05-23', 'resources': {'foo': {'type': 'GenericResourceType'}, 'bar': {'type': 'GenericResourceType', 'depends_on': 'foo'}}})
        stack = parser.Stack(utils.dummy_context(), 'test', tmpl)
        res = stack['bar']
        res.add_explicit_dependencies(self.deps)
        graph = self.deps.graph()
        self.assertIn(res, graph)
        self.assertIn(stack['foo'], graph[res])

    def test_dependson_fail(self):
        tmpl = template.Template({'HeatTemplateFormatVersion': '2012-12-12', 'Resources': {'foo': {'Type': 'GenericResourceType', 'DependsOn': 'wibble'}}})
        stack = parser.Stack(utils.dummy_context(), 'test', tmpl)
        ex = self.assertRaises(exception.InvalidTemplateReference, getattr, stack, 'dependencies')
        self.assertIn('"wibble" (in foo)', str(ex))