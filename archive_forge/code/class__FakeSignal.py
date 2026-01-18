class _FakeSignal(object):
    """If blinker is unavailable, create a fake class with the same

        interface that allows sending of signals but will fail with an
        error on anything else.  Instead of doing anything on send, it
        will just ignore the arguments and do nothing instead.
        """

    def __init__(self, name, doc=None):
        self.name = name
        self.__doc__ = doc

    def _fail(self, *args, **kwargs):
        raise RuntimeError('signalling support is unavailable because the blinker library is not installed.')
    send = lambda *a, **kw: None
    connect = disconnect = has_receivers_for = receivers_for = temporarily_connected_to = connected_to = _fail
    del _fail