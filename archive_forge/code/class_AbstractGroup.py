from typing import Type
from twisted.internet import error
from twisted.internet.protocol import Protocol, connectionDone
from twisted.persisted import styles
from twisted.python.failure import Failure
from twisted.python.reflect import prefixedMethods
from twisted.words.im.locals import OFFLINE, OfflineError
class AbstractGroup:

    def __init__(self, name, account):
        self.name = name
        self.account = account

    def getGroupCommands(self):
        """finds group commands

        these commands are methods on me that start with imgroup_; they are
        called with no arguments
        """
        return prefixedMethods(self, 'imgroup_')

    def getTargetCommands(self, target):
        """finds group commands

        these commands are methods on me that start with imgroup_; they are
        called with a user present within this room as an argument

        you may want to override this in your group in order to filter for
        appropriate commands on the given user
        """
        return prefixedMethods(self, 'imtarget_')

    def join(self):
        if not self.account.client:
            raise OfflineError
        self.account.client.joinGroup(self.name)

    def leave(self):
        if not self.account.client:
            raise OfflineError
        self.account.client.leaveGroup(self.name)

    def __repr__(self) -> str:
        return f'<{self.__class__} {self.name!r}>'

    def __str__(self) -> str:
        return f'{self.name}@{self.account.accountName}'