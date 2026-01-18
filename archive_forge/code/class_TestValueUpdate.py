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
class TestValueUpdate(TestValue):
    scenarios = [('boolean-to-number', dict(param1=True, param_type1='boolean', param2=-100.999, param_type2='number')), ('number-to-string', dict(param1=-77, param_type1='number', param2='mellors', param_type2='string')), ('string-to-map', dict(param1='mellors', param_type1='string', param2={'3': 'turbo'}, param_type2='json')), ('map-to-boolean', dict(param1={'hey': 'there'}, param_type1='json', param2=False, param_type2='boolean')), ('list-to-boolean', dict(param1=['hey', '!'], param_type1='comma_delimited_list', param2=True, param_type2='boolean'))]

    def test_value_update(self):
        ts1, tl1 = self.get_strict_and_loose_templates(self.param_type1)
        ts2, tl2 = self.get_strict_and_loose_templates(self.param_type2)
        env1 = environment.Environment({'parameters': {'param1': self.param1}})
        env2 = environment.Environment({'parameters': {'param1': self.param2}})
        updates = [(ts1, ts2), (ts1, tl2), (tl1, ts2), (tl1, tl2)]
        updates_other_way = [(b, a) for a, b in updates]
        updates.extend(updates_other_way)
        for t_initial, t_updated in updates:
            if t_initial == ts1 or t_initial == tl1:
                p1, p2, e1, e2 = (self.param1, self.param2, env1, env2)
            else:
                p2, p1, e2, e1 = (self.param1, self.param2, env1, env2)
            stack = self.create_stack(copy.deepcopy(t_initial), env=e1)
            self.assertEqual(p1, stack['my_value2'].FnGetAtt('value'))
            res1_id = stack['my_value'].id
            res2_id = stack['my_value2'].id
            res2_uuid = stack['my_value2'].uuid
            updated_stack = parser.Stack(stack.context, 'updated_stack', template.Template(copy.deepcopy(t_updated), env=e2))
            updated_stack.validate()
            stack.update(updated_stack)
            self.assertEqual(p2, stack['my_value2'].FnGetAtt('value'))
            self.assertEqual(res1_id, stack['my_value'].id)
            self.assertEqual(res2_id, stack['my_value2'].id)
            self.assertEqual(res2_uuid, stack['my_value2'].uuid)