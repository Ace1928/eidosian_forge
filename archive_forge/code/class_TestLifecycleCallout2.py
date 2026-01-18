from unittest import mock
from heat.common import lifecycle_plugin_utils
from heat.engine import lifecycle_plugin
from heat.engine import resources
from heat.tests import common
class TestLifecycleCallout2(lifecycle_plugin.LifecyclePlugin):
    """Sample test class for testing pre-op and post-op work on a stack.

    Different ordinal and increment counters by 2.
    """

    def do_pre_op(self, cnxt, stack, current_stack=None, action=None):
        cnxt.pre_counter_for_unit_test += 2

    def do_post_op(self, cnxt, stack, current_stack=None, action=None, is_stack_failure=False):
        cnxt.post_counter_for_unit_test += 2

    def get_ordinal(self):
        return 101