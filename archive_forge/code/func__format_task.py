import codecs
import io
import os
import sys
import warnings
from ..lazy_import import lazy_import
import time
from breezy import (
from .. import config, osutils, trace
from . import NullProgressView, UIFactory
def _format_task(self, task):
    """Format task-specific parts of progress bar.

        :returns: (text_part, counter_part) both unicode strings.
        """
    if not task.show_count:
        s = ''
    elif task.current_cnt is not None and task.total_cnt is not None:
        s = ' %d/%d' % (task.current_cnt, task.total_cnt)
    elif task.current_cnt is not None:
        s = ' %d' % task.current_cnt
    else:
        s = ''
    t = task
    m = task.msg
    while t._parent_task:
        t = t._parent_task
        if t.msg:
            m = t.msg + ':' + m
    return (m, s)