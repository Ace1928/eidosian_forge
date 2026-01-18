from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import progress_tracker
def TrackerUpdateFunc(tracker, result, unused_status):
    """Updates the progress tracker with the result of the operation.

    Args:
      tracker: The ProgressTracker for the operation.
      result: the operation poll result.
      unused_status: map of stages with key as stage key (string) and value is
        the progress_tracker.Stage.
    """
    messages = GetMessagesModule()
    json_val = encoding.MessageToJson(result.metadata)
    preview_metadata = encoding.JsonToMessage(messages.OperationMetadata, json_val).previewMetadata
    logs = ''
    step = ''
    if preview_metadata is not None:
        logs = preview_metadata.logs
        step = preview_metadata.step
    if logs is not None and step is None:
        poller.detailed_message = 'logs={0} '.format(logs)
    elif logs is not None and step is not None:
        poller.detailed_message = 'logs={0}, step={1} '.format(logs, step)
    tracker.Tick()