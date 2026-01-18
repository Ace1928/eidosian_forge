import inspect
import warnings
import debtcollector
from debtcollector.fixtures import disable
from debtcollector import moves
from debtcollector import removals
from debtcollector import renames
from debtcollector.tests import base as test_base
from debtcollector import updating
class ThingB(object):

    @removals.remove()
    def black_tristars(self):
        pass

    @removals.removed_property
    def green_tristars(self):
        return 'green'

    @green_tristars.setter
    def green_tristars(self, value):
        pass

    @green_tristars.deleter
    def green_tristars(self):
        pass

    @removals.removed_property(message='stop using me')
    def green_blue_tristars(self):
        return 'green-blue'

    @removals.remove(category=PendingDeprecationWarning)
    def blue_tristars(self):
        pass

    @removals.remove()
    @classmethod
    def white_wolf(cls):
        pass

    @removals.remove(category=PendingDeprecationWarning)
    @classmethod
    def yellow_wolf(cls):
        pass

    @removals.remove()
    @staticmethod
    def blue_giant():
        pass

    @removals.remove(category=PendingDeprecationWarning)
    @staticmethod
    def green_giant():
        pass