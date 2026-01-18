from taskflow.listeners import base
class CaptureListener(base.Listener):
    """A listener that captures transitions and saves them locally.

    NOTE(harlowja): this listener is *mainly* useful for testing (where it is
    useful to test the appropriate/expected transitions, produced results...
    occurred after engine running) but it could have other usages as well.

    :ivar values: Captured transitions + details (the result of
                  the :py:meth:`._format_capture` method) are stored into this
                  list (a previous list to append to may be provided using the
                  constructor keyword argument of the same name); by default
                  this stores tuples of the format ``(kind, state, details)``.
    """
    FLOW = 'flow'
    TASK = 'task'
    RETRY = 'retry'

    def __init__(self, engine, task_listen_for=base.DEFAULT_LISTEN_FOR, flow_listen_for=base.DEFAULT_LISTEN_FOR, retry_listen_for=base.DEFAULT_LISTEN_FOR, capture_flow=True, capture_task=True, capture_retry=True, skip_tasks=None, skip_retries=None, skip_flows=None, values=None):
        super(CaptureListener, self).__init__(engine, task_listen_for=task_listen_for, flow_listen_for=flow_listen_for, retry_listen_for=retry_listen_for)
        self._capture_flow = capture_flow
        self._capture_task = capture_task
        self._capture_retry = capture_retry
        self._skip_tasks = _freeze_it(skip_tasks)
        self._skip_flows = _freeze_it(skip_flows)
        self._skip_retries = _freeze_it(skip_retries)
        if values is None:
            self.values = []
        else:
            self.values = values

    @staticmethod
    def _format_capture(kind, state, details):
        """Tweak what is saved according to your desire(s)."""
        return (kind, state, details)

    def _task_receiver(self, state, details):
        if self._capture_task:
            if details['task_name'] not in self._skip_tasks:
                self.values.append(self._format_capture(self.TASK, state, details))

    def _retry_receiver(self, state, details):
        if self._capture_retry:
            if details['retry_name'] not in self._skip_retries:
                self.values.append(self._format_capture(self.RETRY, state, details))

    def _flow_receiver(self, state, details):
        if self._capture_flow:
            if details['flow_name'] not in self._skip_flows:
                self.values.append(self._format_capture(self.FLOW, state, details))