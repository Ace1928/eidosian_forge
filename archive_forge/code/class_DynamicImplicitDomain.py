import argparse
import builtins
import enum
import importlib
import inspect
import io
import logging
import os
import pickle
import ply.lex
import re
import sys
import textwrap
import types
from operator import attrgetter
from pyomo.common.collections import Sequence, Mapping
from pyomo.common.deprecation import (
from pyomo.common.fileutils import import_file
from pyomo.common.formatting import wrap_reStructuredText
from pyomo.common.modeling import NOTSET
class DynamicImplicitDomain(object):
    """Implicit domain that can return a custom domain based on the key.

    This provides a mechanism for managing plugin-like systems, where
    the key specifies a source for additional configuration information.
    For example, given the plugin module,
    ``pyomo/common/tests/config_plugin.py``:

    .. literalinclude:: /../../pyomo/common/tests/config_plugin.py
       :start-at: import

    .. doctest::
       :hide:

       >>> import importlib
       >>> import pyomo.common.fileutils
       >>> from pyomo.common.config import ConfigDict, DynamicImplicitDomain

    .. doctest::

       >>> def _pluginImporter(name, config):
       ...     mod = importlib.import_module(name)
       ...     return mod.get_configuration(config)
       >>> config = ConfigDict()
       >>> config.declare('plugins', ConfigDict(
       ...     implicit=True,
       ...     implicit_domain=DynamicImplicitDomain(_pluginImporter)))
       <pyomo.common.config.ConfigDict object at ...>
       >>> config.plugins['pyomo.common.tests.config_plugin'] = {'key1': 5}
       >>> config.display()
       plugins:
         pyomo.common.tests.config_plugin:
           key1: 5
           key2: '5'

    .. note::

       This initializer is only useful for the :py:class:`ConfigDict`
       ``implicit_domain`` argument (and not for "regular" ``domain``
       arguments)

    Parameters
    ----------
    callback: Callable[[str, object], ConfigBase]
        A callable (function) that is passed the ConfigDict key and
        value, and is expected to return the appropriate Config object
        (ConfigValue, ConfigList, or ConfigDict)

    """

    def __init__(self, callback):
        self.callback = callback

    def __call__(self, key, value):
        return self.callback(key, value)