from unittest import mock
from heat.common import lifecycle_plugin_utils
from heat.engine import lifecycle_plugin
from heat.engine import resources
from heat.tests import common
class TestLifecycleCallout4(lifecycle_plugin.LifecyclePlugin):
    """Sample test class for testing pre-op and post-op work on a stack.

    do_pre_op, do_post_op both throw exception.
    """

    def do_pre_op(self, cnxt, stack, current_stack=None, action=None):
        raise Exception()

    def do_post_op(self, cnxt, stack, current_stack=None, action=None, is_stack_failure=False):
        raise Exception()

    def get_ordinal(self):
        return 103