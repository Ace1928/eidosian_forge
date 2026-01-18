from __future__ import annotations
import warnings
from typing import Sequence
from zope.interface import Attribute, Interface, implementer
from incremental import Version
from twisted.python.compat import nativeString
from twisted.python.components import proxyForInterface
from twisted.python.deprecate import deprecatedModuleAttribute
from twisted.python.reflect import prefixedMethodNames
from twisted.web._responses import FORBIDDEN, NOT_FOUND
from twisted.web.error import UnsupportedMethod
def _computeAllowedMethods(resource):
    """
    Compute the allowed methods on a C{Resource} based on defined render_FOO
    methods. Used when raising C{UnsupportedMethod} but C{Resource} does
    not define C{allowedMethods} attribute.
    """
    allowedMethods = []
    for name in prefixedMethodNames(resource.__class__, 'render_'):
        allowedMethods.append(name.encode('ascii'))
    return allowedMethods