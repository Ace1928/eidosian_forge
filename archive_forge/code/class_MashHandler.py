from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import threading
from googlecloudsdk.core import log
from googlecloudsdk.core.util import keyboard_interrupt
import six
class MashHandler(object):
    """MashHandler only invokes its base handler once.

  On the third attempt, the execution is hard-killed.
  """

    def __init__(self, base_handler):
        self._interrupts = 0
        self._base_handler = base_handler
        self._lock = threading.Lock()

    def __call__(self, signal_number, stack_frame):
        with self._lock:
            self._interrupts += 1
            interrupts = self._interrupts
        if interrupts == 1:
            self._base_handler(signal_number, stack_frame)
        elif interrupts == 3:
            keyboard_interrupt.HandleInterrupt(signal_number, stack_frame)