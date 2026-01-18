import inspect
import os
import threading
from contextlib import contextmanager
from datetime import datetime
from traits import trait_notifiers
class ChangeEventRecorder(object):
    """ A single thread trait change event recorder.

    Parameters
    ----------
    container : MultiThreadRecordContainer
        A container to store the records for each trait change.

    Attributes
    ----------
    container : MultiThreadRecordContainer
        A container to store the records for each trait change.
    indent : int
        Indentation level when reporting chained events.
    """

    def __init__(self, container):
        self.indent = 1
        self.container = container

    def pre_tracer(self, obj, name, old, new, handler):
        """ Record a string representation of the trait change dispatch

        """
        indent = self.indent
        time = datetime.utcnow().isoformat(' ')
        container = self.container
        container.record(ChangeMessageRecord(time=time, indent=indent, name=name, old=old, new=new, class_name=obj.__class__.__name__))
        container.record(CallingMessageRecord(time=time, indent=indent, handler=handler.__name__, source=inspect.getsourcefile(handler)))
        self.indent += 1

    def post_tracer(self, obj, name, old, new, handler, exception=None):
        """ Record a string representation of the trait change return

        """
        time = datetime.utcnow().isoformat(' ')
        self.indent -= 1
        indent = self.indent
        if exception:
            exception_msg = ' [EXCEPTION: {}]'.format(exception)
        else:
            exception_msg = ''
        container = self.container
        container.record(ExitMessageRecord(time=time, indent=indent, handler=handler.__name__, exception=exception_msg))
        if indent == 1:
            container.record(SentinelRecord())