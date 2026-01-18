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
def _register_anytrait(self, object, name, remove):
    """ Registers any 'anytrait' listener.
        """
    handler = self.handler()
    if handler is not Undefined:
        object._on_trait_change(handler, remove=remove, dispatch=self.dispatch, priority=self.priority, target=self._get_target())
    return (object, name)