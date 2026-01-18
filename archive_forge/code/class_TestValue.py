import copy
import json
from heat.common import exception
from heat.common import short_id
from heat.common import template_format
from heat.engine import environment
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
class TestValue(common.HeatTestCase):
    simple_template = "\nheat_template_version: '2016-10-14'\nparameters:\n  param1:\n    type: <the type>\nresources:\n  my_value:\n    type: OS::Heat::Value\n    properties:\n      value: {get_param: param1}\n  my_value2:\n    type: OS::Heat::Value\n    properties:\n      value: {get_attr: [my_value, value]}\noutputs:\n  myout:\n    value: {get_attr: [my_value2, value]}\n"

    def get_strict_and_loose_templates(self, param_type):
        template_loose = template_format.parse(self.simple_template)
        template_loose['parameters']['param1']['type'] = param_type
        template_strict = copy.deepcopy(template_loose)
        template_strict['resources']['my_value']['properties']['type'] = param_type
        template_strict['resources']['my_value2']['properties']['type'] = param_type
        return (template_strict, template_loose)

    def parse_stack(self, templ_obj):
        stack_name = 'test_value_stack_%s' % short_id.generate_id()
        stack = parser.Stack(utils.dummy_context(), stack_name, templ_obj)
        stack.validate()
        stack.store()
        return stack

    def create_stack(self, templ, env=None):
        if isinstance(templ, str):
            return self.create_stack(template_format.parse(templ), env=env)
        if isinstance(templ, dict):
            tmpl_obj = template.Template(templ, env=env)
            return self.create_stack(tmpl_obj)
        assert isinstance(templ, template.Template)
        stack = self.parse_stack(templ)
        self.assertIsNone(stack.create())
        return stack