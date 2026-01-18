class BzrError(Exception):
    """
    Base class for errors raised by breezy.

    Attributes:
      internal_error: if True this was probably caused by a brz bug and
                      should be displayed with a traceback; if False (or
                      absent) this was probably a user or environment error
                      and they don't need the gory details.  (That can be
                      overridden by -Derror on the command line.)

      _fmt: Format string to display the error; this is expanded
            by the instance's dict.
    """
    internal_error = False

    def __init__(self, msg=None, **kwds):
        """Construct a new BzrError.

        There are two alternative forms for constructing these objects.
        Either a preformatted string may be passed, or a set of named
        arguments can be given.  The first is for generic "user" errors which
        are not intended to be caught and so do not need a specific subclass.
        The second case is for use with subclasses that provide a _fmt format
        string to print the arguments.

        Keyword arguments are taken as parameters to the error, which can
        be inserted into the format string template.  It's recommended
        that subclasses override the __init__ method to require specific
        parameters.

        Args:
          msg: If given, this is the literal complete text for the error, not
               subject to expansion. 'msg' is used instead of 'message' because
               python evolved and, in 2.6, forbids the use of 'message'.
        """
        Exception.__init__(self)
        if msg is not None:
            self._preformatted_string = msg
        else:
            self._preformatted_string = None
            for key, value in kwds.items():
                setattr(self, key, value)

    def _format(self):
        s = getattr(self, '_preformatted_string', None)
        if s is not None:
            return s
        err = None
        try:
            fmt = self._get_format_string()
            if fmt:
                d = dict(self.__dict__)
                s = fmt % d
                return s
        except Exception as e:
            err = e
        return 'Unprintable exception %s: dict=%r, fmt=%r, error=%r' % (self.__class__.__name__, self.__dict__, getattr(self, '_fmt', None), err)
    __str__ = _format

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, str(self))

    def _get_format_string(self):
        """Return format string for this exception or None"""
        fmt = getattr(self, '_fmt', None)
        if fmt is not None:
            from breezy.i18n import gettext
            return gettext(fmt)

    def __eq__(self, other):
        if self.__class__ is not other.__class__:
            return NotImplemented
        return self.__dict__ == other.__dict__

    def __hash__(self):
        return id(self)