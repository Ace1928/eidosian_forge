class DNSException(Exception):
    """Abstract base class shared by all dnspython exceptions.

    It supports two basic modes of operation:

    a) Old/compatible mode is used if ``__init__`` was called with
    empty *kwargs*.  In compatible mode all *args* are passed
    to the standard Python Exception class as before and all *args* are
    printed by the standard ``__str__`` implementation.  Class variable
    ``msg`` (or doc string if ``msg`` is ``None``) is returned from ``str()``
    if *args* is empty.

    b) New/parametrized mode is used if ``__init__`` was called with
    non-empty *kwargs*.
    In the new mode *args* must be empty and all kwargs must match
    those set in class variable ``supp_kwargs``. All kwargs are stored inside
    ``self.kwargs`` and used in a new ``__str__`` implementation to construct
    a formatted message based on the ``fmt`` class variable, a ``string``.

    In the simplest case it is enough to override the ``supp_kwargs``
    and ``fmt`` class variables to get nice parametrized messages.
    """
    msg = None
    supp_kwargs = set()
    fmt = None

    def __init__(self, *args, **kwargs):
        self._check_params(*args, **kwargs)
        if kwargs:
            self.kwargs = self._check_kwargs(**kwargs)
            self.msg = str(self)
        else:
            self.kwargs = dict()
        if self.msg is None:
            self.msg = self.__doc__
        if args:
            super(DNSException, self).__init__(*args)
        else:
            super(DNSException, self).__init__(self.msg)

    def _check_params(self, *args, **kwargs):
        """Old exceptions supported only args and not kwargs.

        For sanity we do not allow to mix old and new behavior."""
        if args or kwargs:
            assert bool(args) != bool(kwargs), 'keyword arguments are mutually exclusive with positional args'

    def _check_kwargs(self, **kwargs):
        if kwargs:
            assert set(kwargs.keys()) == self.supp_kwargs, 'following set of keyword args is required: %s' % self.supp_kwargs
        return kwargs

    def _fmt_kwargs(self, **kwargs):
        """Format kwargs before printing them.

        Resulting dictionary has to have keys necessary for str.format call
        on fmt class variable.
        """
        fmtargs = {}
        for kw, data in kwargs.items():
            if isinstance(data, (list, set)):
                fmtargs[kw] = list(map(str, data))
                if len(fmtargs[kw]) == 1:
                    fmtargs[kw] = fmtargs[kw].pop()
            else:
                fmtargs[kw] = data
        return fmtargs

    def __str__(self):
        if self.kwargs and self.fmt:
            fmtargs = self._fmt_kwargs(**self.kwargs)
            return self.fmt.format(**fmtargs)
        else:
            return super(DNSException, self).__str__()