import functools
import inspect
import sys
from pyomo.common import DeveloperError
Class decorator to disable methods before construct is called.

    This decorator should be called to create "Abstract" scalar classes
    that override key methods to raise exceptions.  When the construct()
    method is called, the class instance changes type back to the
    original scalar component and the full class functionality is
    restored.  This prevents most class methods from having to begin with
    "`if not self.parent_component()._constructed: raise RuntimeError`"
    