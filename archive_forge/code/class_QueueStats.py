from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QueueStats(_messages.Message):
    """Statistics for a queue.

  Fields:
    concurrentDispatchesCount: Output only. The number of requests that the
      queue has dispatched but has not received a reply for yet.
    effectiveExecutionRate: Output only. The current maximum number of tasks
      per second executed by the queue. The maximum value of this variable is
      controlled by the RateLimits of the Queue. However, this value could be
      less to avoid overloading the endpoints tasks in the queue are
      targeting.
    executedLastMinuteCount: Output only. The number of tasks that the queue
      has dispatched and received a reply for during the last minute. This
      variable counts both successful and non-successful executions.
    oldestEstimatedArrivalTime: Output only. An estimation of the nearest time
      in the future where a task in the queue is scheduled to be executed.
    tasksCount: Output only. An estimation of the number of tasks in the
      queue, that is, the tasks in the queue that haven't been executed, the
      tasks in the queue which the queue has dispatched but has not yet
      received a reply for, and the failed tasks that the queue is retrying.
  """
    concurrentDispatchesCount = _messages.IntegerField(1)
    effectiveExecutionRate = _messages.FloatField(2)
    executedLastMinuteCount = _messages.IntegerField(3)
    oldestEstimatedArrivalTime = _messages.StringField(4)
    tasksCount = _messages.IntegerField(5)