import sys
import threading
import os
import select
import struct
import fcntl
import errno
import termios
import array
import logging
import atexit
from collections import deque
from datetime import datetime, timedelta
import time
import re
import asyncore
import glob
import locale
import subprocess
class ThreadedNotifier(threading.Thread, Notifier):
    """
    This notifier inherits from threading.Thread for instanciating a separate
    thread, and also inherits from Notifier, because it is a threaded notifier.

    Note that every functionality provided by this class is also provided
    through Notifier class. Moreover Notifier should be considered first because
    it is not threaded and could be easily daemonized.
    """

    def __init__(self, watch_manager, default_proc_fun=None, read_freq=0, threshold=0, timeout=None):
        """
        Initialization, initialize base classes. read_freq, threshold and
        timeout parameters are used when looping.

        @param watch_manager: Watch Manager.
        @type watch_manager: WatchManager instance
        @param default_proc_fun: Default processing method. See base class.
        @type default_proc_fun: instance of ProcessEvent
        @param read_freq: if read_freq == 0, events are read asap,
                          if read_freq is > 0, this thread sleeps
                          max(0, read_freq - (timeout / 1000)) seconds.
        @type read_freq: int
        @param threshold: File descriptor will be read only if the accumulated
                          size to read becomes >= threshold. If != 0, you likely
                          want to use it in combination with an appropriate
                          value set for read_freq because without that you would
                          keep looping without really reading anything and that
                          until the amount of events to read is >= threshold. At
                          least with read_freq you might sleep.
        @type threshold: int
        @param timeout: see read_freq above. If provided, it must be set in
                        milliseconds. See
                        https://docs.python.org/3/library/select.html#select.poll.poll
        @type timeout: int
        """
        threading.Thread.__init__(self)
        self._stop_event = threading.Event()
        Notifier.__init__(self, watch_manager, default_proc_fun, read_freq, threshold, timeout)
        self._pipe = os.pipe()
        self._pollobj.register(self._pipe[0], select.POLLIN)

    def stop(self):
        """
        Stop notifier's loop. Stop notification. Join the thread.
        """
        self._stop_event.set()
        os.write(self._pipe[1], b'stop')
        threading.Thread.join(self)
        Notifier.stop(self)
        self._pollobj.unregister(self._pipe[0])
        os.close(self._pipe[0])
        os.close(self._pipe[1])

    def loop(self):
        """
        Thread's main loop. Don't meant to be called by user directly.
        Call inherited start() method instead.

        Events are read only once time every min(read_freq, timeout)
        seconds at best and only if the size of events to read is >= threshold.
        """
        while not self._stop_event.isSet():
            self.process_events()
            ref_time = time.time()
            if self.check_events():
                self._sleep(ref_time)
                self.read_events()

    def run(self):
        """
        Start thread's loop: read and process events until the method
        stop() is called.
        Never call this method directly, instead call the start() method
        inherited from threading.Thread, which then will call run() in
        its turn.
        """
        self.loop()