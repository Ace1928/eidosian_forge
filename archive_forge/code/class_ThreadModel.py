import traceback
from oslo_reports.models import with_default_views as mwdv
from oslo_reports.views.text import threading as text_views
class ThreadModel(mwdv.ModelWithDefaultViews):
    """A Thread Model

    This model holds data for information about an
    individual thread.  It holds both a thread id,
    as well as a stack trace for the thread

    .. seealso::

        Class :class:`StackTraceModel`

    :param int thread_id: the id of the thread
    :param stack: the python stack state for the current thread
    """

    def __init__(self, thread_id, stack):
        super(ThreadModel, self).__init__(text_view=text_views.ThreadView())
        self['thread_id'] = thread_id
        self['stack_trace'] = StackTraceModel(stack)