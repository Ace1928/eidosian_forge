import builtins
import copy
import inspect
import linecache
import sys
from inspect import getmro
from io import StringIO
from typing import Callable, NoReturn, TypeVar
import opcode
from twisted.python import reflect
@classmethod
def _findFailure(cls):
    """
        Find the failure that represents the exception currently in context.
        """
    tb = sys.exc_info()[-1]
    if not tb:
        return
    secondLastTb = None
    lastTb = tb
    while lastTb.tb_next:
        secondLastTb = lastTb
        lastTb = lastTb.tb_next
    lastFrame = lastTb.tb_frame
    if lastFrame.f_code is cls.raiseException.__code__:
        return lastFrame.f_locals.get('self')
    if not lastFrame.f_code.co_code or lastFrame.f_code.co_code[lastTb.tb_lasti] != cls._yieldOpcode:
        return
    if secondLastTb:
        frame = secondLastTb.tb_frame
        if frame.f_code is cls.throwExceptionIntoGenerator.__code__:
            return frame.f_locals.get('self')
    frame = tb.tb_frame.f_back
    if frame and frame.f_code is cls.throwExceptionIntoGenerator.__code__:
        return frame.f_locals.get('self')