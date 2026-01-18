import importlib
import os
import stat
import subprocess
import sys
import tempfile
import warnings
from .. import osutils, symbol_versioning
class _CompatabilityThunkFeature(Feature):
    """This feature is just a thunk to another feature.

    It issues a deprecation warning if it is accessed, to let you know that you
    should really use a different feature.
    """

    def __init__(self, dep_version, module, name, replacement_name, replacement_module=None):
        super().__init__()
        self._module = module
        if replacement_module is None:
            replacement_module = module
        self._replacement_module = replacement_module
        self._name = name
        self._replacement_name = replacement_name
        self._dep_version = dep_version
        self._feature = None

    def _ensure(self):
        if self._feature is None:
            from breezy import pyutils
            depr_msg = self._dep_version % ('%s.%s' % (self._module, self._name))
            use_msg = ' Use {}.{} instead.'.format(self._replacement_module, self._replacement_name)
            symbol_versioning.warn(depr_msg + use_msg, DeprecationWarning, stacklevel=5)
            self._feature = pyutils.get_named_object(self._replacement_module, self._replacement_name)

    def _probe(self):
        self._ensure()
        return self._feature._probe()