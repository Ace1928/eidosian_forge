import sys
import twisted.internet
from twisted.test.test_twisted import SetAsideModule
class NoReactor(SetAsideModule):
    """
    Context manager that uninstalls the reactor, if any, and then restores it
    afterwards.
    """

    def __init__(self):
        SetAsideModule.__init__(self, 'twisted.internet.reactor')

    def __enter__(self):
        SetAsideModule.__enter__(self)
        if 'twisted.internet.reactor' in self.modules:
            del twisted.internet.reactor

    def __exit__(self, excType, excValue, traceback):
        SetAsideModule.__exit__(self, excType, excValue, traceback)
        reactor = self.modules.get('twisted.internet.reactor', None)
        if reactor is not None:
            twisted.internet.reactor = reactor
        else:
            try:
                del twisted.internet.reactor
            except AttributeError:
                pass