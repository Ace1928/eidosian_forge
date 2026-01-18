from typing import Type
from twisted.internet import error
from twisted.internet.protocol import Protocol, connectionDone
from twisted.persisted import styles
from twisted.python.failure import Failure
from twisted.python.reflect import prefixedMethods
from twisted.words.im.locals import OFFLINE, OfflineError
class AbstractPerson:

    def __init__(self, name, baseAccount):
        self.name = name
        self.account = baseAccount
        self.status = OFFLINE

    def getPersonCommands(self):
        """finds person commands

        these commands are methods on me that start with imperson_; they are
        called with no arguments
        """
        return prefixedMethods(self, 'imperson_')

    def getIdleTime(self):
        """
        Returns a string.
        """
        return '--'

    def __repr__(self) -> str:
        return f'<{self.__class__} {self.name!r}/{self.status}>'

    def __str__(self) -> str:
        return f'{self.name}@{self.account.accountName}'