import ctypes
import threading
from ..ports import BaseInput, BaseOutput, sleep
from . import portmidi_init as pm
def _thread_main(self):
    try:
        while not self._stop_event:
            self._receive()
            for message in self._parser:
                if self.callback:
                    self.callback(message)
            sleep()
    finally:
        if self._stop_event:
            self._stop_event.set()