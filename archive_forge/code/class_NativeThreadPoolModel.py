import futurist
from oslo_log import log as logging
from glance.i18n import _LE
class NativeThreadPoolModel(ThreadPoolModel):
    """A ThreadPoolModel suitable for use with native threads."""
    DEFAULTSIZE = 16

    @staticmethod
    def get_threadpool_executor_class():
        return futurist.ThreadPoolExecutor