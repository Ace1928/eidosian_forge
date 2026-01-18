import collections
import copy
import inspect
import logging
import pkgutil
import sys
import types
class DecoratorData:
    """Decorator data.

    Attributes
    ----------
    patches : list of gorilla.Patch
        Patches created through the decorators.
    override : dict
        Any overriding value defined by the :func:`destination`, :func:`name`,
        and :func:`settings` decorators.
    filter : bool or None
        Value defined by the :func:`filter` decorator, if any, or ``None``
        otherwise.
    """

    def __init__(self):
        """Constructor."""
        self.patches = []
        self.override = {}
        self.filter = None