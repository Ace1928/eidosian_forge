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
class ListenerBase:
    """
    Abstract base class for both ListenerItem and ListenerGroup.
    """

    def set_notify(self, notify):
        """ Set notify state on this listener.

        Parameters
        ----------
        notify : bool
            True if this listener should notify, else False.
        """
        raise NotImplementedError

    def set_next(self, next):
        """ Set the child listener for this listener.

        Parameters
        ----------
        next : ListenerBase
            The next level (if any) of ListenerBase object to be called when
            any of our listened to traits is changed:
        """
        raise NotImplementedError

    def register(self, new):
        """ Registers new listeners.
        """
        raise NotImplementedError

    def unregister(self, old):
        """ Unregisters any existing listeners.
        """
        raise NotImplementedError

    def handle(self, object, name, old, new):
        """ Handles a trait change for a simple trait.
        """
        raise NotImplementedError

    def handle_list(self, object, name, old, new):
        """ Handles a trait change for a list trait.
        """
        raise NotImplementedError

    def handle_list_items(self, object, name, old, new):
        """ Handles a trait change for a list traits items.
        """
        raise NotImplementedError

    def handle_dict(self, object, name, old, new):
        """ Handles a trait change for a dictionary trait.
        """
        raise NotImplementedError

    def handle_dict_items(self, object, name, old, new):
        """ Handles a trait change for a dictionary traits items.
        """
        raise NotImplementedError