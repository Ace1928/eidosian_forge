import sys
class catch_warnings(object):
    """A context manager that copies and restores the warnings filter upon
    exiting the context.

    The 'record' argument specifies whether warnings should be captured by a
    custom implementation of warnings.showwarning() and be appended to a list
    returned by the context manager. Otherwise None is returned by the context
    manager. The objects appended to the list are arguments whose attributes
    mirror the arguments to showwarning().

    The 'module' argument is to specify an alternative module to the module
    named 'warnings' and imported under that name. This argument is only useful
    when testing the warnings module itself.

    If the 'action' argument is not None, the remaining arguments are passed
    to warnings.simplefilter() as if it were called immediately on entering the
    context.
    """

    def __init__(self, *, record=False, module=None, action=None, category=Warning, lineno=0, append=False):
        """Specify whether to record warnings and if an alternative module
        should be used other than sys.modules['warnings'].

        For compatibility with Python 3.0, please consider all arguments to be
        keyword-only.

        """
        self._record = record
        self._module = sys.modules['warnings'] if module is None else module
        self._entered = False
        if action is None:
            self._filter = None
        else:
            self._filter = (action, category, lineno, append)

    def __repr__(self):
        args = []
        if self._record:
            args.append('record=True')
        if self._module is not sys.modules['warnings']:
            args.append('module=%r' % self._module)
        name = type(self).__name__
        return '%s(%s)' % (name, ', '.join(args))

    def __enter__(self):
        if self._entered:
            raise RuntimeError('Cannot enter %r twice' % self)
        self._entered = True
        self._filters = self._module.filters
        self._module.filters = self._filters[:]
        self._module._filters_mutated()
        self._showwarning = self._module.showwarning
        self._showwarnmsg_impl = self._module._showwarnmsg_impl
        if self._filter is not None:
            simplefilter(*self._filter)
        if self._record:
            log = []
            self._module._showwarnmsg_impl = log.append
            self._module.showwarning = self._module._showwarning_orig
            return log
        else:
            return None

    def __exit__(self, *exc_info):
        if not self._entered:
            raise RuntimeError('Cannot exit %r without entering first' % self)
        self._module.filters = self._filters
        self._module._filters_mutated()
        self._module.showwarning = self._showwarning
        self._module._showwarnmsg_impl = self._showwarnmsg_impl