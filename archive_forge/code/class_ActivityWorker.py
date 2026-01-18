import time
from functools import wraps
from boto.swf.layer1 import Layer1
from boto.swf.layer1_decisions import Layer1Decisions
class ActivityWorker(Actor):
    """Base class for SimpleWorkflow activity workers."""

    @wraps(Layer1.respond_activity_task_canceled)
    def cancel(self, task_token=None, details=None):
        """RespondActivityTaskCanceled."""
        if task_token is None:
            task_token = self.last_tasktoken
        return self._swf.respond_activity_task_canceled(task_token, details)

    @wraps(Layer1.respond_activity_task_completed)
    def complete(self, task_token=None, result=None):
        """RespondActivityTaskCompleted."""
        if task_token is None:
            task_token = self.last_tasktoken
        return self._swf.respond_activity_task_completed(task_token, result)

    @wraps(Layer1.respond_activity_task_failed)
    def fail(self, task_token=None, details=None, reason=None):
        """RespondActivityTaskFailed."""
        if task_token is None:
            task_token = self.last_tasktoken
        return self._swf.respond_activity_task_failed(task_token, details, reason)

    @wraps(Layer1.record_activity_task_heartbeat)
    def heartbeat(self, task_token=None, details=None):
        """RecordActivityTaskHeartbeat."""
        if task_token is None:
            task_token = self.last_tasktoken
        return self._swf.record_activity_task_heartbeat(task_token, details)

    @wraps(Layer1.poll_for_activity_task)
    def poll(self, **kwargs):
        """PollForActivityTask."""
        task_list = self.task_list
        if 'task_list' in kwargs:
            task_list = kwargs.get('task_list')
            del kwargs['task_list']
        task = self._swf.poll_for_activity_task(self.domain, task_list, **kwargs)
        self.last_tasktoken = task.get('taskToken')
        return task