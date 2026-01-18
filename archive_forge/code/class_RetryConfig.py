from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RetryConfig(_messages.Message):
    """Retry config. These settings determine when a failed task attempt is
  retried.

  Fields:
    maxAttempts: Number of attempts per task. Cloud Tasks will attempt the
      task `max_attempts` times (that is, if the first attempt fails, then
      there will be `max_attempts - 1` retries). Must be >= -1. If unspecified
      when the queue is created, Cloud Tasks will pick the default. -1
      indicates unlimited attempts. This field has the same meaning as
      [task_retry_limit in queue.yaml/xml](https://cloud.google.com/appengine/
      docs/standard/python/config/queueref#retry_parameters).
    maxBackoff: A task will be scheduled for retry between min_backoff and
      max_backoff duration after it fails, if the queue's RetryConfig
      specifies that the task should be retried. If unspecified when the queue
      is created, Cloud Tasks will pick the default. The value must be given
      as a string that indicates the length of time (in seconds) followed by
      `s` (for "seconds"). For more information on the format, see the
      documentation for [Duration](https://protobuf.dev/reference/protobuf/goo
      gle.protobuf/#duration). `max_backoff` will be truncated to the nearest
      second. This field has the same meaning as [max_backoff_seconds in queue
      .yaml/xml](https://cloud.google.com/appengine/docs/standard/python/confi
      g/queueref#retry_parameters).
    maxDoublings: The time between retries will double `max_doublings` times.
      A task's retry interval starts at min_backoff, then doubles
      `max_doublings` times, then increases linearly, and finally retries at
      intervals of max_backoff up to max_attempts times. For example, if
      min_backoff is 10s, max_backoff is 300s, and `max_doublings` is 3, then
      the a task will first be retried in 10s. The retry interval will double
      three times, and then increase linearly by 2^3 * 10s. Finally, the task
      will retry at intervals of max_backoff until the task has been attempted
      max_attempts times. Thus, the requests will retry at 10s, 20s, 40s, 80s,
      160s, 240s, 300s, 300s, .... If unspecified when the queue is created,
      Cloud Tasks will pick the default. This field has the same meaning as
      [max_doublings in queue.yaml/xml](https://cloud.google.com/appengine/doc
      s/standard/python/config/queueref#retry_parameters).
    maxRetryDuration: If positive, `max_retry_duration` specifies the time
      limit for retrying a failed task, measured from when the task was first
      attempted. Once `max_retry_duration` time has passed *and* the task has
      been attempted max_attempts times, no further attempts will be made and
      the task will be deleted. If zero, then the task age is unlimited. If
      unspecified when the queue is created, Cloud Tasks will pick the
      default. The value must be given as a string that indicates the length
      of time (in seconds) followed by `s` (for "seconds"). For the maximum
      possible value or the format, see the documentation for [Duration](https
      ://protobuf.dev/reference/protobuf/google.protobuf/#duration).
      `max_retry_duration` will be truncated to the nearest second. This field
      has the same meaning as [task_age_limit in queue.yaml/xml](https://cloud
      .google.com/appengine/docs/standard/python/config/queueref#retry_paramet
      ers).
    minBackoff: A task will be scheduled for retry between min_backoff and
      max_backoff duration after it fails, if the queue's RetryConfig
      specifies that the task should be retried. If unspecified when the queue
      is created, Cloud Tasks will pick the default. The value must be given
      as a string that indicates the length of time (in seconds) followed by
      `s` (for "seconds"). For more information on the format, see the
      documentation for [Duration](https://protobuf.dev/reference/protobuf/goo
      gle.protobuf/#duration). `min_backoff` will be truncated to the nearest
      second. This field has the same meaning as [min_backoff_seconds in queue
      .yaml/xml](https://cloud.google.com/appengine/docs/standard/python/confi
      g/queueref#retry_parameters).
  """
    maxAttempts = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    maxBackoff = _messages.StringField(2)
    maxDoublings = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    maxRetryDuration = _messages.StringField(4)
    minBackoff = _messages.StringField(5)