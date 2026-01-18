from heat.db import api as db_api
from heat.engine import service
from heat.engine import stack
from heat.tests.convergence.framework import message_processor
from heat.tests.convergence.framework import message_queue
from heat.tests.convergence.framework import scenario_template
from heat.tests import utils
class SynchronousThreadGroupManager(service.ThreadGroupManager):
    """Wrapper for thread group manager.

    The start method of thread group manager needs to be overridden to
    run the function synchronously so the convergence scenario
    tests can be run.
    """

    def start(self, stack_id, func, *args, **kwargs):
        func(*args, **kwargs)