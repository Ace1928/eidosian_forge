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
class _IEncodingResource(Interface):
    """
    A resource which knows about L{_IRequestEncoderFactory}.

    @since: 12.3
    """

    def getEncoder(request):
        """
        Parse the request and return an encoder if applicable, using
        L{_IRequestEncoderFactory.encoderForRequest}.

        @return: A L{_IRequestEncoder}, or L{None}.
        """