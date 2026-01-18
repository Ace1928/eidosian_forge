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
@implementer(_IEncodingResource)
class EncodingResourceWrapper(proxyForInterface(IResource)):
    """
    Wrap a L{IResource}, potentially applying an encoding to the response body
    generated.

    Note that the returned children resources won't be wrapped, so you have to
    explicitly wrap them if you want the encoding to be applied.

    @ivar encoders: A list of
        L{_IRequestEncoderFactory<twisted.web.iweb._IRequestEncoderFactory>}
        returning L{_IRequestEncoder<twisted.web.iweb._IRequestEncoder>} that
        may transform the data passed to C{Request.write}. The list must be
        sorted in order of priority: the first encoder factory handling the
        request will prevent the others from doing the same.
    @type encoders: C{list}.

    @since: 12.3
    """

    def __init__(self, original, encoders):
        super().__init__(original)
        self._encoders = encoders

    def getEncoder(self, request):
        """
        Browser the list of encoders looking for one applicable encoder.
        """
        for encoderFactory in self._encoders:
            encoder = encoderFactory.encoderForRequest(request)
            if encoder is not None:
                return encoder