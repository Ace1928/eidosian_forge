from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class StepTypeValueValuesEnum(_messages.Enum):
    """Output only. The type of the step this step entry belongs to.

    Values:
      STEP_TYPE_UNSPECIFIED: Invalid step type.
      STEP_ASSIGN: The step entry assigns new variable(s).
      STEP_STD_LIB_CALL: The step entry calls a standard library routine.
      STEP_CONNECTOR_CALL: The step entry calls a connector.
      STEP_SUBWORKFLOW_CALL: The step entry calls a subworklfow.
      STEP_CALL: The step entry calls a subworkflow/stdlib.
      STEP_SWITCH: The step entry executes a switch-case block.
      STEP_CONDITION: The step entry executes a condition inside a switch.
      STEP_FOR: The step entry executes a for loop.
      STEP_FOR_ITERATION: The step entry executes a iteration of a for loop.
      STEP_PARALLEL_FOR: The step entry executes a parallel for loop.
      STEP_PARALLEL_BRANCH: The step entry executes a series of parallel
        branch(es).
      STEP_PARALLEL_BRANCH_ENTRY: The step entry executes a branch of a
        parallel branch.
      STEP_TRY_RETRY_EXCEPT: The step entry executes a try/retry/except block.
      STEP_TRY: The step entry executes the try part of a try/retry/except
        block.
      STEP_RETRY: The step entry executes the retry part of a try/retry/except
        block.
      STEP_EXCEPT: The step entry executes the except part of a
        try/retry/except block.
      STEP_RETURN: The step entry returns.
      STEP_RAISE: The step entry raises an error.
      STEP_GOTO: The step entry jumps to another step.
    """
    STEP_TYPE_UNSPECIFIED = 0
    STEP_ASSIGN = 1
    STEP_STD_LIB_CALL = 2
    STEP_CONNECTOR_CALL = 3
    STEP_SUBWORKFLOW_CALL = 4
    STEP_CALL = 5
    STEP_SWITCH = 6
    STEP_CONDITION = 7
    STEP_FOR = 8
    STEP_FOR_ITERATION = 9
    STEP_PARALLEL_FOR = 10
    STEP_PARALLEL_BRANCH = 11
    STEP_PARALLEL_BRANCH_ENTRY = 12
    STEP_TRY_RETRY_EXCEPT = 13
    STEP_TRY = 14
    STEP_RETRY = 15
    STEP_EXCEPT = 16
    STEP_RETURN = 17
    STEP_RAISE = 18
    STEP_GOTO = 19