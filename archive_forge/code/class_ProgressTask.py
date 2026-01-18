import os
import time
class ProgressTask:
    """Model component of a progress indicator.

    Most code that needs to indicate progress should update one of these,
    and it will in turn update the display, if one is present.

    Code updating the task may also set fields as hints about how to display
    it: show_pct, show_spinner, show_eta, show_count, show_bar.  UIs
    will not necessarily respect all these fields.

    The message given when updating a task must be unicode, not bytes.

    Attributes:
      update_latency: The interval (in seconds) at which the PB should be
        updated.  Setting this to zero suggests every update should be shown
        synchronously.

      show_transport_activity: If true (default), transport activity
        will be shown when this task is drawn.  Disable it if you're sure
        that only irrelevant or uninteresting transport activity can occur
        during this task.
    """

    def __init__(self, parent_task=None, ui_factory=None, progress_view=None):
        """Construct a new progress task.

        Args:
          parent_task: Enclosing ProgressTask or None.
          progress_view: ProgressView to display this ProgressTask.
          ui_factory: The UI factory that will display updates;
            deprecated in favor of passing progress_view directly.

        Normally you should not call this directly but rather through
        `ui_factory.nested_progress_bar`.
        """
        self._parent_task = parent_task
        self._last_update = 0
        self.total_cnt = None
        self.current_cnt = None
        self.msg = ''
        self.ui_factory = ui_factory
        self.progress_view = progress_view
        self.show_pct = False
        self.show_spinner = True
        self.show_eta = (False,)
        self.show_count = True
        self.show_bar = True
        self.update_latency = 0.1
        self.show_transport_activity = True

    def __repr__(self):
        return '{}({!r}/{!r}, msg={!r})'.format(self.__class__.__name__, self.current_cnt, self.total_cnt, self.msg)

    def update(self, msg, current_cnt=None, total_cnt=None):
        """Report updated task message and if relevent progress counters

        The message given must be unicode, not a byte string.
        """
        self.msg = msg
        self.current_cnt = current_cnt
        if total_cnt:
            self.total_cnt = total_cnt
        if self.progress_view:
            self.progress_view.show_progress(self)
        else:
            self.ui_factory._progress_updated(self)

    def tick(self):
        self.update(self.msg)

    def finished(self):
        if self.progress_view:
            self.progress_view.task_finished(self)
        else:
            self.ui_factory._progress_finished(self)

    def make_sub_task(self):
        return ProgressTask(self, ui_factory=self.ui_factory, progress_view=self.progress_view)

    def _overall_completion_fraction(self, child_fraction=0.0):
        """Return fractional completion of this task and its parents

        Returns None if no completion can be computed."""
        if self.current_cnt is not None and self.total_cnt:
            own_fraction = (float(self.current_cnt) + child_fraction) / self.total_cnt
        else:
            own_fraction = child_fraction
        if self._parent_task is None:
            return own_fraction
        else:
            if own_fraction is None:
                own_fraction = 0.0
            return self._parent_task._overall_completion_fraction(own_fraction)

    def clear(self):
        if self.progress_view:
            self.progress_view.clear()
        else:
            self.ui_factory.clear_term()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finished()
        return False