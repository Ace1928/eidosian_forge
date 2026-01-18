import futurist
from taskflow.conductors.backends import impl_executor
@staticmethod
def _executor_factory():
    return futurist.SynchronousExecutor()