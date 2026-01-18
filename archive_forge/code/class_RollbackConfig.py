from .. import errors
from ..constants import IS_WINDOWS_PLATFORM
from ..utils import (
class RollbackConfig(UpdateConfig):
    """
    Used to specify the way container rollbacks should be performed by a
    service

    Args:
        parallelism (int): Maximum number of tasks to be rolled back in one
          iteration (0 means unlimited parallelism). Default: 0
        delay (int): Amount of time between rollbacks, in nanoseconds.
        failure_action (string): Action to take if a rolled back task fails to
          run, or stops running during the rollback. Acceptable values are
          ``continue``, ``pause`` or ``rollback``.
          Default: ``continue``
        monitor (int): Amount of time to monitor each rolled back task for
          failures, in nanoseconds.
        max_failure_ratio (float): The fraction of tasks that may fail during
          a rollback before the failure action is invoked, specified as a
          floating point number between 0 and 1. Default: 0
        order (string): Specifies the order of operations when rolling out a
          rolled back task. Either ``start-first`` or ``stop-first`` are
          accepted.
    """
    pass