import os
import sys
from typing import TYPE_CHECKING, Dict
def _trace_func(frame, event, arg):
    if event == 'call':
        indent = ''
        for i in range(_trace_depth):
            _trace_frame(thread, frame, indent)
            indent += '  '
            frame = frame.f_back
            if not frame:
                break
    elif event == 'exception':
        exception, value, traceback = arg
        print('First chance exception raised:', repr(exception))