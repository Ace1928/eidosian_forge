import warnings
from twisted.internet import reactor
class wxRunner:
    """Make sure GUI events are handled."""

    def __init__(self, app):
        self.app = app

    def run(self):
        """
        Execute pending WX events followed by WX idle events and
        reschedule.
        """
        while self.app.Pending():
            self.app.Dispatch()
        self.app.ProcessIdle()
        reactor.callLater(0.02, self.run)