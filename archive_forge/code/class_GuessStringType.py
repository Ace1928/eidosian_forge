import ctypes
import functools
from winappdbg import compat
import sys
class GuessStringType(object):
    """
    Decorator that guesses the correct version (A or W) to call
    based on the types of the strings passed as parameters.

    Calls the B{ANSI} version if the only string types are ANSI.

    Calls the B{Unicode} version if Unicode or mixed string types are passed.

    The default if no string arguments are passed depends on the value of the
    L{t_default} class variable.

    @type fn_ansi: function
    @ivar fn_ansi: ANSI version of the API function to call.
    @type fn_unicode: function
    @ivar fn_unicode: Unicode (wide) version of the API function to call.

    @type t_default: type
    @cvar t_default: Default string type to use.
        Possible values are:
         - type('') for ANSI
         - type(u'') for Unicode
    """
    t_ansi = type('')
    t_unicode = type(u'')
    t_default = t_ansi

    def __init__(self, fn_ansi, fn_unicode):
        """
        @type  fn_ansi: function
        @param fn_ansi: ANSI version of the API function to call.
        @type  fn_unicode: function
        @param fn_unicode: Unicode (wide) version of the API function to call.
        """
        self.fn_ansi = fn_ansi
        self.fn_unicode = fn_unicode
        try:
            self.__name__ = self.fn_ansi.__name__[:-1]
        except AttributeError:
            pass
        try:
            self.__module__ = self.fn_ansi.__module__
        except AttributeError:
            pass
        try:
            self.__doc__ = self.fn_ansi.__doc__
        except AttributeError:
            pass

    def __call__(self, *argv, **argd):
        t_ansi = self.t_ansi
        v_types = [type(item) for item in argv]
        v_types.extend([type(value) for key, value in compat.iteritems(argd)])
        if self.t_default == t_ansi:
            fn = self.fn_ansi
        else:
            fn = self.fn_unicode
        if self.t_unicode in v_types:
            if t_ansi in v_types:
                argv = list(argv)
                for index in compat.xrange(len(argv)):
                    if v_types[index] == t_ansi:
                        argv[index] = compat.unicode(argv[index])
                for key, value in argd.items():
                    if type(value) == t_ansi:
                        argd[key] = compat.unicode(value)
            fn = self.fn_unicode
        elif t_ansi in v_types:
            fn = self.fn_ansi
        return fn(*argv, **argd)