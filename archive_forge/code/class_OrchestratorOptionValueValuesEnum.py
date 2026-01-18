from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class OrchestratorOptionValueValuesEnum(_messages.Enum):
    """The option of whether running each test within its own invocation of
    instrumentation with Android Test Orchestrator or not. ** Orchestrator is
    only compatible with AndroidJUnitRunner version 1.1 or higher! **
    Orchestrator offers the following benefits: - No shared state - Crashes
    are isolated - Logs are scoped per test See for more information about
    Android Test Orchestrator. If not set, the test will be run without the
    orchestrator.

    Values:
      ORCHESTRATOR_OPTION_UNSPECIFIED: Default value: the server will choose
        the mode. Currently implies that the test will run without the
        orchestrator. In the future, all instrumentation tests will be run
        with the orchestrator. Using the orchestrator is highly encouraged
        because of all the benefits it offers.
      USE_ORCHESTRATOR: Run test using orchestrator. ** Only compatible with
        AndroidJUnitRunner version 1.1 or higher! ** Recommended.
      DO_NOT_USE_ORCHESTRATOR: Run test without using orchestrator.
    """
    ORCHESTRATOR_OPTION_UNSPECIFIED = 0
    USE_ORCHESTRATOR = 1
    DO_NOT_USE_ORCHESTRATOR = 2