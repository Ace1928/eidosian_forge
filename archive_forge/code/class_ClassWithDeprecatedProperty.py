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
class ClassWithDeprecatedProperty:
    """
    Class with a single deprecated property.
    """
    _someProtectedValue = None

    @deprecatedProperty(Version('Twisted', 1, 2, 3))
    def someProperty(self):
        """
        Getter docstring.

        @return: The property.
        """
        return self._someProtectedValue

    @someProperty.setter
    def someProperty(self, value):
        """
        Setter docstring.
        """
        self._someProtectedValue = value