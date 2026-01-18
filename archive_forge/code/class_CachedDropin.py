import os
import pickle
import sys
import types
from typing import Iterable, Optional, Type, TypeVar
from zope.interface import Interface, providedBy
from twisted.python import log
from twisted.python.components import getAdapterFactory
from twisted.python.modules import getModule
from twisted.python.reflect import namedAny
class CachedDropin:
    """
    A collection of L{CachedPlugin} instances from a particular module in a
    plugin package.

    @type moduleName: C{str}
    @ivar moduleName: The fully qualified name of the plugin module this
        represents.

    @type description: C{str} or L{None}
    @ivar description: A brief explanation of this collection of plugins
        (probably the plugin module's docstring).

    @type plugins: C{list}
    @ivar plugins: The L{CachedPlugin} instances which were loaded from this
        dropin.
    """

    def __init__(self, moduleName, description):
        self.moduleName = moduleName
        self.description = description
        self.plugins = []