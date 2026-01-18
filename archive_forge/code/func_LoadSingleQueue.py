from __future__ import absolute_import
from __future__ import unicode_literals
import os
def LoadSingleQueue(queue_info, open_fn=None):
    """Loads a `queue.yaml` file/string and returns a `QueueInfoExternal` object.

  Args:
    queue_info: The contents of a `queue.yaml` file, as a string.
    open_fn: Function for opening files. Unused.

  Returns:
    A `QueueInfoExternal` object.
  """
    builder = yaml_object.ObjectBuilder(QueueInfoExternal)
    handler = yaml_builder.BuilderHandler(builder)
    listener = yaml_listener.EventListener(handler)
    listener.Parse(queue_info)
    queue_info = handler.GetResults()
    if len(queue_info) < 1:
        raise MalformedQueueConfiguration('Empty queue configuration.')
    if len(queue_info) > 1:
        raise MalformedQueueConfiguration('Multiple queue: sections in configuration.')
    return queue_info[0]