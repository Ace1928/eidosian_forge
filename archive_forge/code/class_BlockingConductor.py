import futurist
from taskflow.conductors.backends import impl_executor
class BlockingConductor(impl_executor.ExecutorConductor):
    """Blocking conductor that processes job(s) in a blocking manner."""
    MAX_SIMULTANEOUS_JOBS = 1
    '\n    Default maximum number of jobs that can be in progress at the same time.\n    '

    @staticmethod
    def _executor_factory():
        return futurist.SynchronousExecutor()

    def __init__(self, name, jobboard, persistence=None, engine=None, engine_options=None, wait_timeout=None, log=None, max_simultaneous_jobs=MAX_SIMULTANEOUS_JOBS):
        super(BlockingConductor, self).__init__(name, jobboard, persistence=persistence, engine=engine, engine_options=engine_options, wait_timeout=wait_timeout, log=log, max_simultaneous_jobs=max_simultaneous_jobs)