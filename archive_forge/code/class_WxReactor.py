from queue import Empty, Queue
from twisted.internet import _threadedselect
from twisted.python import log, runtime
class WxReactor(_threadedselect.ThreadedSelectReactor):
    """
    wxPython reactor.

    wxPython drives the event loop, select() runs in a thread.
    """
    _stopping = False

    def registerWxApp(self, wxapp):
        """
        Register wxApp instance with the reactor.
        """
        self.wxapp = wxapp

    def _installSignalHandlersAgain(self):
        """
        wx sometimes removes our own signal handlers, so re-add them.
        """
        try:
            import signal
            signal.signal(signal.SIGINT, signal.default_int_handler)
        except ImportError:
            return
        self._signals.install()

    def stop(self):
        """
        Stop the reactor.
        """
        if self._stopping:
            return
        self._stopping = True
        _threadedselect.ThreadedSelectReactor.stop(self)

    def _runInMainThread(self, f):
        """
        Schedule function to run in main wx/Twisted thread.

        Called by the select() thread.
        """
        if hasattr(self, 'wxapp'):
            wxCallAfter(f)
        else:
            self._postQueue.put(f)

    def _stopWx(self):
        """
        Stop the wx event loop if it hasn't already been stopped.

        Called during Twisted event loop shutdown.
        """
        if hasattr(self, 'wxapp'):
            self.wxapp.ExitMainLoop()

    def run(self, installSignalHandlers=True):
        """
        Start the reactor.
        """
        self._postQueue = Queue()
        if not hasattr(self, 'wxapp'):
            log.msg('registerWxApp() was not called on reactor, registering my own wxApp instance.')
            self.registerWxApp(wxPySimpleApp())
        self.interleave(self._runInMainThread, installSignalHandlers=installSignalHandlers)
        if installSignalHandlers:
            self.callLater(0, self._installSignalHandlersAgain)
        self.addSystemEventTrigger('after', 'shutdown', self._stopWx)
        self.addSystemEventTrigger('after', 'shutdown', lambda: self._postQueue.put(None))
        if runtime.platform.isMacOSX():
            t = ProcessEventsTimer(self.wxapp)
            t.Start(2)
        self.wxapp.MainLoop()
        wxapp = self.wxapp
        del self.wxapp
        if not self._stopping:
            self.stop()
            wxapp.ProcessPendingEvents()
            while 1:
                try:
                    f = self._postQueue.get(timeout=0.01)
                except Empty:
                    continue
                else:
                    if f is None:
                        break
                    try:
                        f()
                    except BaseException:
                        log.err()