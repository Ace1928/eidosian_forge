import threading
import time
from twisted.python import log
class Watchdog(threading.Thread):
    """
    Watch a given thread, call a list of functions when that thread exits.
    """

    def __init__(self, canary, shutdown_function):
        threading.Thread.__init__(self, name='CrochetShutdownWatchdog')
        self._canary = canary
        self._shutdown_function = shutdown_function

    def run(self):
        while self._canary.is_alive():
            time.sleep(0.1)
        self._shutdown_function()