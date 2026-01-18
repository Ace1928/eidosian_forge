from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StartEnvironmentMetadata(_messages.Message):
    """Message included in the metadata field of operations returned from
  StartEnvironment.

  Enums:
    StateValueValuesEnum: Current state of the environment being started.

  Fields:
    state: Current state of the environment being started.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Current state of the environment being started.

    Values:
      STATE_UNSPECIFIED: The environment's start state is unknown.
      STARTING: The environment is in the process of being started, but no
        additional details are available.
      UNARCHIVING_DISK: Startup is waiting for the user's disk to be
        unarchived. This can happen when the user returns to Cloud Shell after
        not having used it for a while, and suggests that startup will take
        longer than normal.
      AWAITING_COMPUTE_RESOURCES: Startup is waiting for compute resources to
        be assigned to the environment. This should normally happen very
        quickly, but an environment might stay in this state for an extended
        period of time if the system is experiencing heavy load.
      FINISHED: Startup has completed. If the start operation was successful,
        the user should be able to establish an SSH connection to their
        environment. Otherwise, the operation will contain details of the
        failure.
    """
        STATE_UNSPECIFIED = 0
        STARTING = 1
        UNARCHIVING_DISK = 2
        AWAITING_COMPUTE_RESOURCES = 3
        FINISHED = 4
    state = _messages.EnumField('StateValueValuesEnum', 1)