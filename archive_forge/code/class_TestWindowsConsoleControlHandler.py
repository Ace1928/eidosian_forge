import sys
import time
from functools import wraps
from pytest import mark
from zmq.tests import BaseZMQTestCase
from zmq.utils.win32 import allow_interrupt
@mark.new_console
class TestWindowsConsoleControlHandler(BaseZMQTestCase):

    @mark.new_console
    @mark.skipif(not sys.platform.startswith('win'), reason='Windows only test')
    def test_handler(self):

        @count_calls
        def interrupt_polling():
            print('Caught CTRL-C!')
        from ctypes import windll
        from ctypes.wintypes import BOOL, DWORD
        kernel32 = windll.LoadLibrary('kernel32')
        GenerateConsoleCtrlEvent = kernel32.GenerateConsoleCtrlEvent
        GenerateConsoleCtrlEvent.argtypes = (DWORD, DWORD)
        GenerateConsoleCtrlEvent.restype = BOOL
        try:
            with allow_interrupt(interrupt_polling) as context:
                result = GenerateConsoleCtrlEvent(0, 0)
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        else:
            if result == 0:
                raise OSError()
            else:
                self.fail('Expecting `KeyboardInterrupt` exception!')
        assert interrupt_polling.__calls__ == 1