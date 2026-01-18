import re
import string
import weakref
from string import whitespace
from types import MethodType
from .constants import DefaultValue
from .trait_base import Undefined, Uninitialized
from .trait_errors import TraitError
from .trait_notifiers import TraitChangeNotifyWrapper
from .util.weakiddict import WeakIDKeyDict
class ListenerGroup(ListenerBase):
    """
    Listener description for a collection of items.

    The ListenerParser produces a ListenerGroup rather than a ListenerItem
    when parsing strings like ``[abc,def]``.

    Parameters
    ----------
    items : list
        List of ListenerItem objects representing the components of the group.
    """

    def __init__(self, *, items):
        self.items = items
        self.next = None

    def __repr__(self, seen=None):
        """Returns a string representation of the object.

        Since the object graph may have cycles, we extend the basic __repr__
        API to include a set of objects we've already seen while constructing
        a string representation. When this method tries to get the repr of
        a ListenerItem or ListenerGroup, we will use the extended API and build
        up the set of seen objects. The repr of a seen object will just be
        '<cycle>'.
        """
        if seen is None:
            seen = set()
        seen.add(self)
        lines = ['%s(items = [' % self.__class__.__name__]
        for item in self.items:
            lines.extend(indent(item.__repr__(seen), True).split('\n'))
            lines[-1] += ','
        lines.append('])')
        return '\n'.join(lines)

    def set_notify(self, notify):
        """ Set notify state on this listener.

        Parameters
        ----------
        notify : bool
            True if this listener should notify, else False.
        """
        for item in self.items:
            item.set_notify(notify)

    def set_next(self, next):
        """ Set the child listener for this one.

        Parameters
        ----------
        next : ListenerBase
            The next level (if any) of ListenerBase object to be called when
            any of our listened to traits is changed:
        """
        for item in self.items:
            item.set_next(next)
        self.next = next if self.items else None

    def register(self, new):
        """ Registers new listeners.
        """
        for item in self.items:
            item.register(new)
        return INVALID_DESTINATION

    def unregister(self, old):
        """ Unregisters any existing listeners.
        """
        for item in self.items:
            item.unregister(old)