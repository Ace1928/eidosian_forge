import os
import sys
import time
import threading
import traceback
from paste.util.classinstance import classinstancemethod
class JythonMonitor(Monitor):
    """
            Monitor that utilizes Jython's special
            ``_systemrestart.SystemRestart`` exception.

            When raised from the main thread it causes Jython to reload
            the interpreter in the existing Java process (avoiding
            startup time).

            Note that this functionality of Jython is experimental and
            may change in the future.
            """

    def periodic_reload(self):
        while True:
            if not self.check_reload():
                raise SystemRestart()
            time.sleep(self.poll_interval)