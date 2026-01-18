import inspect
import warnings
import debtcollector
from debtcollector.fixtures import disable
from debtcollector import moves
from debtcollector import removals
from debtcollector import renames
from debtcollector.tests import base as test_base
from debtcollector import updating
class KittyKat(object):

    @moves.moved_method('supermeow')
    def meow(self, volume=11):
        return self.supermeow(volume)

    @moves.moved_method('supermeow', category=PendingDeprecationWarning)
    def maow(self, volume=11):
        return self.supermeow(volume)

    def supermeow(self, volume=11):
        return 'supermeow'