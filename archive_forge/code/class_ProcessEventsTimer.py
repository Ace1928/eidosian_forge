from queue import Empty, Queue
from twisted.internet import _threadedselect
from twisted.python import log, runtime
class ProcessEventsTimer(wxTimer):
    """
    Timer that tells wx to process pending events.

    This is necessary on macOS, probably due to a bug in wx, if we want
    wxCallAfters to be handled when modal dialogs, menus, etc.  are open.
    """

    def __init__(self, wxapp):
        wxTimer.__init__(self)
        self.wxapp = wxapp

    def Notify(self):
        """
        Called repeatedly by wx event loop.
        """
        self.wxapp.ProcessPendingEvents()