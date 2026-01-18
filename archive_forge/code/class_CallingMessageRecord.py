import inspect
import os
import threading
from contextlib import contextmanager
from datetime import datetime
from traits import trait_notifiers
class CallingMessageRecord(object):
    """ Message record for a change handler call.

    """
    __slots__ = ('time', 'indent', 'handler', 'source')

    def __init__(self, time, indent, handler, source):
        self.time = time
        self.indent = indent
        self.handler = handler
        self.source = source

    def __str__(self):
        gap = self.indent * 2 + SPACES_TO_ALIGN_WITH_CHANGE_MESSAGE
        return CALLINGMSG.format(time=self.time, action='CALLING', handler=self.handler, source=self.source, gap=gap)