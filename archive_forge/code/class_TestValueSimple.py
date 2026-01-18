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
class TestValueSimple(TestValue):
    scenarios = [('boolean', dict(param1=True, param_type='boolean')), ('list', dict(param1=['a', 'b', 'Z'], param_type='comma_delimited_list')), ('map', dict(param1={'a': 'Z', 'B': 'y'}, param_type='json')), ('number-int', dict(param1=-11, param_type='number')), ('number-float', dict(param1=100.999, param_type='number')), ('string', dict(param1='Perchance to dream', param_type='string'))]

    def test_value(self):
        ts, tl = self.get_strict_and_loose_templates(self.param_type)
        env = environment.Environment({'parameters': {'param1': self.param1}})
        for templ_dict in [ts, tl]:
            stack = self.create_stack(templ_dict, env)
            self.assertEqual(self.param1, stack['my_value'].FnGetAtt('value'))
            self.assertEqual(self.param1, stack['my_value2'].FnGetAtt('value'))
            stack._update_all_resource_data(False, True)
            self.assertEqual(self.param1, stack.outputs['myout'].get_value())