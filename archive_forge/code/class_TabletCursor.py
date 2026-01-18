import sys
import enum
import warnings
import operator
from pyglet.event import EventDispatcher
class TabletCursor:
    """A distinct cursor used on a tablet.

    Most tablets support at least a *stylus* and an *erasor* cursor; this
    object is used to distinguish them when tablet events are generated.

    :Ivariables:
        `name` : str
            Name of the cursor.
    """

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, self.name)