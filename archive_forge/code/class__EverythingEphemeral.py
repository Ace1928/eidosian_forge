import os
import pickle
import sys
from zope.interface import Interface, implementer
from twisted.persisted import styles
from twisted.python import log, runtime
class _EverythingEphemeral(styles.Ephemeral):
    initRun = 0

    def __init__(self, mainMod):
        """
        @param mainMod: The '__main__' module that this class will proxy.
        """
        self.mainMod = mainMod

    def __getattr__(self, key):
        try:
            return getattr(self.mainMod, key)
        except AttributeError:
            if self.initRun:
                raise
            else:
                log.msg('Warning!  Loading from __main__: %s' % key)
                return styles.Ephemeral()