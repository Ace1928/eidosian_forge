import copy as _copy
import os as _os
import re as _re
import sys as _sys
import textwrap as _textwrap
from gettext import gettext as _
class FileType(object):
    """Factory for creating file object types

    Instances of FileType are typically passed as type= arguments to the
    ArgumentParser add_argument() method.

    Keyword Arguments:
        - mode -- A string indicating how the file is to be opened. Accepts the
            same values as the builtin open() function.
        - bufsize -- The file's desired buffer size. Accepts the same values as
            the builtin open() function.
    """

    def __init__(self, mode='r', bufsize=None):
        self._mode = mode
        self._bufsize = bufsize

    def __call__(self, string):
        if string == '-':
            if 'r' in self._mode:
                return _sys.stdin
            elif 'w' in self._mode:
                return _sys.stdout
            else:
                msg = _('argument "-" with mode %r' % self._mode)
                raise ValueError(msg)
        try:
            if self._bufsize:
                return open(string, self._mode, self._bufsize)
            else:
                return open(string, self._mode)
        except IOError:
            err = _sys.exc_info()[1]
            message = _("can't open '%s': %s")
            raise ArgumentTypeError(message % (string, err))

    def __repr__(self):
        args = [self._mode, self._bufsize]
        args_str = ', '.join([repr(arg) for arg in args if arg is not None])
        return '%s(%s)' % (type(self).__name__, args_str)