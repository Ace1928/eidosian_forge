from taskflow import exceptions as exc
from taskflow import states
from taskflow import test
class CheckFlowTransitionTest(TransitionTest):

    def setUp(self):
        super(CheckFlowTransitionTest, self).setUp()
        self.check_transition = states.check_flow_transition
        self.transition_exc_regexp = '^Flow transition.*not allowed'

    def test_to_same_state(self):
        self.assertTransitionIgnored(states.SUCCESS, states.SUCCESS)

    def test_rerunning_allowed(self):
        self.assertTransitionAllowed(states.SUCCESS, states.RUNNING)

    def test_no_resuming_from_pending(self):
        self.assertTransitionIgnored(states.PENDING, states.RESUMING)

    def test_resuming_from_running(self):
        self.assertTransitionAllowed(states.RUNNING, states.RESUMING)

    def test_bad_transition_raises(self):
        self.assertTransitionForbidden(states.FAILURE, states.SUCCESS)