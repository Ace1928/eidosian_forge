from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExecutionTriggerValueValuesEnum(_messages.Enum):
    """Job execution trigger.

    Values:
      EXECUTION_TRIGGER_UNSPECIFIED: The job execution trigger is unspecified.
      TASK_CONFIG: The job was triggered by Dataplex based on trigger spec
        from task definition.
      RUN_REQUEST: The job was triggered by the explicit call of Task API.
    """
    EXECUTION_TRIGGER_UNSPECIFIED = 0
    TASK_CONFIG = 1
    RUN_REQUEST = 2