from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class TestStateValueValuesEnum(_messages.Enum):
    """The current rolled-up state of the test matrix. If this state is
    already final, then the cancelation request will have no effect.

    Values:
      TEST_STATE_UNSPECIFIED: Do not use. For proto versioning only.
      VALIDATING: The execution or matrix is being validated.
      PENDING: The execution or matrix is waiting for resources to become
        available.
      RUNNING: The execution is currently being processed. Can only be set on
        an execution.
      FINISHED: The execution or matrix has terminated normally. On a matrix
        this means that the matrix level processing completed normally, but
        individual executions may be in an ERROR state.
      ERROR: The execution or matrix has stopped because it encountered an
        infrastructure failure.
      UNSUPPORTED_ENVIRONMENT: The execution was not run because it
        corresponds to a unsupported environment. Can only be set on an
        execution.
      INCOMPATIBLE_ENVIRONMENT: The execution was not run because the provided
        inputs are incompatible with the requested environment. Example:
        requested AndroidVersion is lower than APK's minSdkVersion Can only be
        set on an execution.
      INCOMPATIBLE_ARCHITECTURE: The execution was not run because the
        provided inputs are incompatible with the requested architecture.
        Example: requested device does not support running the native code in
        the supplied APK Can only be set on an execution.
      CANCELLED: The user cancelled the execution. Can only be set on an
        execution.
      INVALID: The execution or matrix was not run because the provided inputs
        are not valid. Examples: input file is not of the expected type, is
        malformed/corrupt, or was flagged as malware
    """
    TEST_STATE_UNSPECIFIED = 0
    VALIDATING = 1
    PENDING = 2
    RUNNING = 3
    FINISHED = 4
    ERROR = 5
    UNSUPPORTED_ENVIRONMENT = 6
    INCOMPATIBLE_ENVIRONMENT = 7
    INCOMPATIBLE_ARCHITECTURE = 8
    CANCELLED = 9
    INVALID = 10