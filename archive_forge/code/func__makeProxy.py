import inspect
import sys
import types
import warnings
from os.path import normcase
from warnings import catch_warnings, simplefilter
from incremental import Version
from twisted.python import deprecate
from twisted.python.deprecate import (
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.python.test import deprecatedattributes
from twisted.python.test.modules_helpers import TwistedModulesMixin
from twisted.trial.unittest import SynchronousTestCase
from twisted.python.deprecate import deprecatedModuleAttribute
from incremental import Version
from twisted.python import deprecate
from twisted.python import deprecate
def _makeProxy(self, **attrs):
    """
        Create a temporary module proxy object.

        @param **kw: Attributes to initialise on the temporary module object

        @rtype: L{twistd.python.deprecate._ModuleProxy}
        """
    mod = types.ModuleType('foo')
    for key, value in attrs.items():
        setattr(mod, key, value)
    return deprecate._ModuleProxy(mod)