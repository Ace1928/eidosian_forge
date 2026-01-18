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
class TestValueLessSimple(TestValue):
    template_bad = "\nheat_template_version: '2016-10-14'\nparameters:\n  param1:\n    type: json\nresources:\n  my_value:\n    type: OS::Heat::Value\n    properties:\n      value: {get_param: param1}\n      type: number\n"
    template_map = "\nheat_template_version: '2016-10-14'\nparameters:\n  param1:\n    type: json\n  param2:\n    type: json\nresources:\n  my_value:\n    type: OS::Heat::Value\n    properties:\n      value: {get_param: param1}\n      type: json\n  my_value2:\n    type: OS::Heat::Value\n    properties:\n      value: {map_merge: [{get_attr: [my_value, value]}, {get_param: param2}]}\n      type: json\n"
    template_yaql = "\nheat_template_version: '2016-10-14'\nparameters:\n  param1:\n    type: number\n  param2:\n    type: comma_delimited_list\nresources:\n  my_value:\n    type: OS::Heat::Value\n    properties:\n      value: {get_param: param1}\n      type: number\n  my_value2:\n    type: OS::Heat::Value\n    properties:\n      value:\n        yaql:\n          expression: $.data.param2.select(int($)).min()\n          data:\n            param2: {get_param: param2}\n      type: number\n  my_value3:\n    type: OS::Heat::Value\n    properties:\n      value:\n        yaql:\n          expression: min($.data.v1,$.data.v2)\n          data:\n            v1: {get_attr: [my_value, value]}\n            v2: {get_attr: [my_value2, value]}\n"

    def test_validation_fail(self):
        param1 = {'one': 'croissant'}
        env = environment.Environment({'parameters': {'param1': json.dumps(param1)}})
        self.assertRaises(exception.StackValidationFailed, self.create_stack, self.template_bad, env)

    def test_map(self):
        param1 = {'one': 'skipper', 'two': 'antennae'}
        param2 = {'one': 'monarch', 'three': 'sky'}
        env = environment.Environment({'parameters': {'param1': json.dumps(param1), 'param2': json.dumps(param2)}})
        stack = self.create_stack(self.template_map, env)
        my_value = stack['my_value']
        self.assertEqual(param1, my_value.FnGetAtt('value'))
        my_value2 = stack['my_value2']
        self.assertEqual({'one': 'monarch', 'two': 'antennae', 'three': 'sky'}, my_value2.FnGetAtt('value'))

    def test_yaql(self):
        param1 = -800
        param2 = [-8, 0, 4, -11, 2]
        env = environment.Environment({'parameters': {'param1': param1, 'param2': param2}})
        stack = self.create_stack(self.template_yaql, env)
        my_value = stack['my_value']
        self.assertEqual(param1, my_value.FnGetAtt('value'))
        my_value2 = stack['my_value2']
        self.assertEqual(min(param2), my_value2.FnGetAtt('value'))
        my_value3 = stack['my_value3']
        self.assertEqual(param1, my_value3.FnGetAtt('value'))