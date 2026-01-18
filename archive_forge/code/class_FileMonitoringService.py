import os
import warnings
from zope.interface import implementer
from twisted.application import internet, service
from twisted.cred.portal import Portal
from twisted.internet import defer
from twisted.mail import protocols, smtp
from twisted.mail.interfaces import IAliasableDomain, IDomain
from twisted.python import log, util
class FileMonitoringService(internet.TimerService):
    """
    A service for monitoring changes to files.

    @type files: L{list} of L{list} of (E{1}) L{float}, (E{2}) L{bytes},
        (E{3}) callable which takes a L{bytes} argument, (E{4}) L{float}
    @ivar files: Information about files to be monitored.  Each list entry
        provides the following information for a file: interval in seconds
        between checks, filename, callback function, time of last modification
        to the file.

    @type intervals: L{_IntervalDifferentialIterator
        <twisted.python.util._IntervalDifferentialIterator>}
    @ivar intervals: Intervals between successive file checks.

    @type _call: L{IDelayedCall <twisted.internet.interfaces.IDelayedCall>}
        provider
    @ivar _call: The next scheduled call to check a file.

    @type index: L{int}
    @ivar index: The index of the next file to be checked.
    """

    def __init__(self):
        """
        Initialize the file monitoring service.
        """
        self.files = []
        self.intervals = iter(util.IntervalDifferential([], 60))

    def startService(self):
        """
        Start the file monitoring service.
        """
        service.Service.startService(self)
        self._setupMonitor()

    def _setupMonitor(self):
        """
        Schedule the next monitoring call.
        """
        from twisted.internet import reactor
        t, self.index = self.intervals.next()
        self._call = reactor.callLater(t, self._monitor)

    def stopService(self):
        """
        Stop the file monitoring service.
        """
        service.Service.stopService(self)
        if self._call:
            self._call.cancel()
            self._call = None

    def monitorFile(self, name, callback, interval=10):
        """
        Start monitoring a file for changes.

        @type name: L{bytes}
        @param name: The name of a file to monitor.

        @type callback: callable which takes a L{bytes} argument
        @param callback: The function to call when the file has changed.

        @type interval: L{float}
        @param interval: The interval in seconds between checks.
        """
        try:
            mtime = os.path.getmtime(name)
        except BaseException:
            mtime = 0
        self.files.append([interval, name, callback, mtime])
        self.intervals.addInterval(interval)

    def unmonitorFile(self, name):
        """
        Stop monitoring a file.

        @type name: L{bytes}
        @param name: A file name.
        """
        for i in range(len(self.files)):
            if name == self.files[i][1]:
                self.intervals.removeInterval(self.files[i][0])
                del self.files[i]
                break

    def _monitor(self):
        """
        Monitor a file and make a callback if it has changed.
        """
        self._call = None
        if self.index is not None:
            name, callback, mtime = self.files[self.index][1:]
            try:
                now = os.path.getmtime(name)
            except BaseException:
                now = 0
            if now > mtime:
                log.msg(f'{name} changed, notifying listener')
                self.files[self.index][3] = now
                callback(name)
        self._setupMonitor()