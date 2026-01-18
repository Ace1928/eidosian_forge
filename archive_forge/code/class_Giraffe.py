import inspect
import warnings
import debtcollector
from debtcollector.fixtures import disable
from debtcollector import moves
from debtcollector import removals
from debtcollector import renames
from debtcollector.tests import base as test_base
from debtcollector import updating
class Giraffe(object):
    color = 'orange'
    colour = moves.moved_read_only_property('colour', 'color')

    @property
    def height(self):
        return 2
    heightt = moves.moved_read_only_property('heightt', 'height')