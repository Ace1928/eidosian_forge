import builtins
import types
import sys
from inspect import getmembers
from webob.exc import HTTPFound
from .util import iscontroller, _cfg
class HookControllerMeta(type):
    """
    A base class for controllers that would like to specify hooks on
    their controller methods. Simply create a list of hook objects
    called ``__hooks__`` as a member of the controller's namespace.
    """

    def __init__(cls, name, bases, dict_):
        hooks = set(dict_.get('__hooks__', []))
        for base in bases:
            for hook in getattr(base, '__hooks__', []):
                hooks.add(hook)
        walk_controller(cls, cls, hooks)