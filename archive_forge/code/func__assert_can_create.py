from heat.common import template_format
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def _assert_can_create(self, templ):
    stack = parser.Stack(self.ctx, utils.random_name(), template.Template(templ))
    stack.store()
    stack.create()
    self.assertEqual((parser.Stack.CREATE, parser.Stack.COMPLETE), stack.state)
    return stack