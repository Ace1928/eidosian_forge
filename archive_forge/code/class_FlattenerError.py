from collections.abc import Sequence
from typing import Optional, Union, cast
from twisted.python.compat import nativeString
from twisted.web._responses import RESPONSES
class FlattenerError(Exception):
    """
    An error occurred while flattening an object.

    @ivar _roots: A list of the objects on the flattener's stack at the time
        the unflattenable object was encountered.  The first element is least
        deeply nested object and the last element is the most deeply nested.
    """

    def __init__(self, exception, roots, traceback):
        self._exception = exception
        self._roots = roots
        self._traceback = traceback
        Exception.__init__(self, exception, roots, traceback)

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

    def __repr__(self) -> str:
        """
        Present a string representation which includes a template traceback, so
        we can tell where this error occurred in the template, as well as in
        Python.
        """
        from traceback import format_list
        if self._roots:
            roots = '  ' + '\n  '.join([self._formatRoot(r) for r in self._roots]) + '\n'
        else:
            roots = ''
        if self._traceback:
            traceback = '\n'.join([line for entry in format_list(self._traceback) for line in entry.splitlines()]) + '\n'
        else:
            traceback = ''
        return cast(str, 'Exception while flattening:\n' + roots + traceback + self._exception.__class__.__name__ + ': ' + str(self._exception) + '\n')

    def __str__(self) -> str:
        return repr(self)