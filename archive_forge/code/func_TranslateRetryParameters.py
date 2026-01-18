from __future__ import absolute_import
from __future__ import unicode_literals
import os
def TranslateRetryParameters(retry):
    """Populates a `TaskQueueRetryParameters` from a `queueinfo.RetryParameters`.

  Args:
    retry: A `queueinfo.RetryParameters` that is read from `queue.yaml` that
        describes the queue's retry parameters.

  Returns:
    A `taskqueue_service_pb.TaskQueueRetryParameters` proto populated with the
    data from `retry`.

  Raises:
    MalformedQueueConfiguration: If the retry parameters are invalid.
  """
    params = taskqueue_service_pb.TaskQueueRetryParameters()
    if retry.task_retry_limit is not None:
        params.set_retry_limit(int(retry.task_retry_limit))
    if retry.task_age_limit is not None:
        params.set_age_limit_sec(ParseTaskAgeLimit(retry.task_age_limit))
    if retry.min_backoff_seconds is not None:
        params.set_min_backoff_sec(float(retry.min_backoff_seconds))
    if retry.max_backoff_seconds is not None:
        params.set_max_backoff_sec(float(retry.max_backoff_seconds))
    if retry.max_doublings is not None:
        params.set_max_doublings(int(retry.max_doublings))
    if params.has_min_backoff_sec() and (not params.has_max_backoff_sec()):
        if params.min_backoff_sec() > params.max_backoff_sec():
            params.set_max_backoff_sec(params.min_backoff_sec())
    if not params.has_min_backoff_sec() and params.has_max_backoff_sec():
        if params.min_backoff_sec() > params.max_backoff_sec():
            params.set_min_backoff_sec(params.max_backoff_sec())
    if params.has_retry_limit() and params.retry_limit() < 0:
        raise MalformedQueueConfiguration('Task retry limit must not be less than zero.')
    if params.has_age_limit_sec() and (not params.age_limit_sec() > 0):
        raise MalformedQueueConfiguration('Task age limit must be greater than zero.')
    if params.has_min_backoff_sec() and params.min_backoff_sec() < 0:
        raise MalformedQueueConfiguration('Min backoff seconds must not be less than zero.')
    if params.has_max_backoff_sec() and params.max_backoff_sec() < 0:
        raise MalformedQueueConfiguration('Max backoff seconds must not be less than zero.')
    if params.has_max_doublings() and params.max_doublings() < 0:
        raise MalformedQueueConfiguration('Max doublings must not be less than zero.')
    if params.has_min_backoff_sec() and params.has_max_backoff_sec() and (params.min_backoff_sec() > params.max_backoff_sec()):
        raise MalformedQueueConfiguration('Min backoff sec must not be greater than than max backoff sec.')
    return params