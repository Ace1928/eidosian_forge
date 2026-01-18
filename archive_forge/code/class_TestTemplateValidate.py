import copy
import hashlib
import json
import fixtures
from stevedore import extension
from heat.common import exception
from heat.common import template_format
from heat.engine.cfn import functions as cfn_funcs
from heat.engine.cfn import parameters as cfn_p
from heat.engine.cfn import template as cfn_t
from heat.engine.clients.os import nova
from heat.engine import environment
from heat.engine import function
from heat.engine.hot import template as hot_t
from heat.engine import node_data
from heat.engine import rsrc_defn
from heat.engine import stack
from heat.engine import stk_defn
from heat.engine import template
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
class TestTemplateValidate(common.HeatTestCase):

    def test_template_validate_cfn_check_t_digest(self):
        t = {'AWSTemplateFormatVersion': '2010-09-09', 'Description': 'foo', 'Parameters': {}, 'Mappings': {}, 'Resources': {'server': {'Type': 'OS::Nova::Server'}}, 'Outputs': {}}
        tmpl = template.Template(t)
        self.assertIsNone(tmpl.t_digest)
        tmpl.validate()
        self.assertEqual(hashlib.sha256(str(t).encode('utf-8')).hexdigest(), tmpl.t_digest, 'invalid template digest')

    def test_template_validate_cfn_good(self):
        t = {'AWSTemplateFormatVersion': '2010-09-09', 'Description': 'foo', 'Parameters': {}, 'Mappings': {}, 'Resources': {'server': {'Type': 'OS::Nova::Server'}}, 'Outputs': {}}
        tmpl = template.Template(t)
        err = tmpl.validate()
        self.assertIsNone(err)
        t = {'HeatTemplateFormatVersion': '2012-12-12', 'Description': 'foo', 'Parameters': {}, 'Mappings': {}, 'Resources': {'server': {'Type': 'OS::Nova::Server'}}, 'Outputs': {}}
        tmpl = template.Template(t)
        err = tmpl.validate()
        self.assertIsNone(err)

    def test_template_validate_cfn_bad_section(self):
        t = {'AWSTemplateFormatVersion': '2010-09-09', 'Description': 'foo', 'Parameteers': {}, 'Mappings': {}, 'Resources': {'server': {'Type': 'OS::Nova::Server'}}, 'Outputs': {}}
        tmpl = template.Template(t)
        err = self.assertRaises(exception.InvalidTemplateSection, tmpl.validate)
        self.assertIn('Parameteers', str(err))

    def test_template_validate_cfn_empty(self):
        t = template_format.parse('\n            AWSTemplateFormatVersion: 2010-09-09\n            Parameters:\n            Resources:\n            Outputs:\n            ')
        tmpl = template.Template(t)
        err = tmpl.validate()
        self.assertIsNone(err)

    def test_get_resources_good(self):
        """Test get resources successful."""
        t = template_format.parse('\n            AWSTemplateFormatVersion: 2010-09-09\n            Resources:\n              resource1:\n                Type: AWS::EC2::Instance\n                Properties:\n                  property1: value1\n                Metadata:\n                  foo: bar\n                DependsOn: dummy\n                DeletionPolicy: dummy\n                UpdatePolicy:\n                  foo: bar\n        ')
        expected = {'resource1': {'Type': 'AWS::EC2::Instance', 'Properties': {'property1': 'value1'}, 'Metadata': {'foo': 'bar'}, 'DependsOn': 'dummy', 'DeletionPolicy': 'dummy', 'UpdatePolicy': {'foo': 'bar'}}}
        tmpl = template.Template(t)
        self.assertEqual(expected, tmpl[tmpl.RESOURCES])

    def test_get_resources_bad_no_data(self):
        """Test get resources without any mapping."""
        t = template_format.parse('\n            AWSTemplateFormatVersion: 2010-09-09\n            Resources:\n              resource1:\n        ')
        tmpl = template.Template(t)
        error = self.assertRaises(exception.StackValidationFailed, tmpl.validate)
        self.assertEqual('Each Resource must contain a Type key.', str(error))

    def test_get_resources_no_type(self):
        """Test get resources with invalid key."""
        t = template_format.parse('\n            AWSTemplateFormatVersion: 2010-09-09\n            Resources:\n              resource1:\n                Properties:\n                  property1: value1\n                Metadata:\n                  foo: bar\n                DependsOn: dummy\n                DeletionPolicy: dummy\n                UpdatePolicy:\n                  foo: bar\n        ')
        tmpl = template.Template(t)
        error = self.assertRaises(exception.StackValidationFailed, tmpl.validate)
        self.assertEqual('Each Resource must contain a Type key.', str(error))

    def test_template_validate_hot_check_t_digest(self):
        t = {'heat_template_version': '2015-04-30', 'description': 'foo', 'parameters': {}, 'resources': {'server': {'type': 'OS::Nova::Server'}}, 'outputs': {}}
        tmpl = template.Template(t)
        self.assertIsNone(tmpl.t_digest)
        tmpl.validate()
        self.assertEqual(hashlib.sha256(str(t).encode('utf-8')).hexdigest(), tmpl.t_digest, 'invalid template digest')

    def test_template_validate_hot_good(self):
        t = {'heat_template_version': '2013-05-23', 'description': 'foo', 'parameters': {}, 'resources': {'server': {'type': 'OS::Nova::Server'}}, 'outputs': {}}
        tmpl = template.Template(t)
        err = tmpl.validate()
        self.assertIsNone(err)

    def test_template_validate_hot_bad_section(self):
        t = {'heat_template_version': '2013-05-23', 'description': 'foo', 'parameteers': {}, 'resources': {'server': {'type': 'OS::Nova::Server'}}, 'outputs': {}}
        tmpl = template.Template(t)
        err = self.assertRaises(exception.InvalidTemplateSection, tmpl.validate)
        self.assertIn('parameteers', str(err))