import copy
import io
from unittest import mock
from osc_lib import exceptions as exc
from osc_lib import utils
import testscenarios
import yaml
from heatclient.common import template_format
from heatclient import exc as heat_exc
from heatclient.osc.v1 import stack
from heatclient.tests import inline_templates
from heatclient.tests.unit.osc.v1 import fakes as orchestration_fakes
from heatclient.v1 import events
from heatclient.v1 import resources
from heatclient.v1 import stacks
class TestStackResume(_TestStackCheckBase, TestStack):

    def setUp(self):
        super(TestStackResume, self).setUp()
        self._setUp(stack.ResumeStack(self.app, None), self.mock_client.actions.resume, 'RESUME')

    def test_stack_resume(self):
        self._test_stack_action()

    def test_stack_resume_multi(self):
        self._test_stack_action_multi()

    def test_stack_resume_wait(self):
        self._test_stack_action_wait()

    def test_stack_resume_wait_error(self):
        self._test_stack_action_wait_error()

    def test_stack_resume_exception(self):
        self._test_stack_action_exception()