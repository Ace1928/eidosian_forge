from collections.abc import Sequence
from typing import Optional, Union, cast
from twisted.python.compat import nativeString
from twisted.web._responses import RESPONSES
def _formatRoot(self, obj):
    """
        Convert an object from C{self._roots} to a string suitable for
        inclusion in a render-traceback (like a normal Python traceback, but
        can include "frame" source locations which are not in Python source
        files).

        @param obj: Any object which can be a render step I{root}.
            Typically, L{Tag}s, strings, and other simple Python types.

        @return: A string representation of C{obj}.
        @rtype: L{str}
        """
    from twisted.web.template import Tag
    if isinstance(obj, (bytes, str)):
        if len(obj) > 40:
            if isinstance(obj, str):
                ellipsis = '<...>'
            else:
                ellipsis = b'<...>'
            return ascii(obj[:20] + ellipsis + obj[-20:])
        else:
            return ascii(obj)
    elif isinstance(obj, Tag):
        if obj.filename is None:
            return 'Tag <' + obj.tagName + '>'
        else:
            return 'File "%s", line %d, column %d, in "%s"' % (obj.filename, obj.lineNumber, obj.columnNumber, obj.tagName)
    else:
        return ascii(obj)