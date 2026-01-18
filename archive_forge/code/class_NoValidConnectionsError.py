import socket
class NoValidConnectionsError(socket.error):
    """
    Multiple connection attempts were made and no families succeeded.

    This exception class wraps multiple "real" underlying connection errors,
    all of which represent failed connection attempts. Because these errors are
    not guaranteed to all be of the same error type (i.e. different errno,
    `socket.error` subclass, message, etc) we expose a single unified error
    message and a ``None`` errno so that instances of this class match most
    normal handling of `socket.error` objects.

    To see the wrapped exception objects, access the ``errors`` attribute.
    ``errors`` is a dict whose keys are address tuples (e.g. ``('127.0.0.1',
    22)``) and whose values are the exception encountered trying to connect to
    that address.

    It is implied/assumed that all the errors given to a single instance of
    this class are from connecting to the same hostname + port (and thus that
    the differences are in the resolution of the hostname - e.g. IPv4 vs v6).

    .. versionadded:: 1.16
    """

    def __init__(self, errors):
        """
        :param dict errors:
            The errors dict to store, as described by class docstring.
        """
        addrs = sorted(errors.keys())
        body = ', '.join([x[0] for x in addrs[:-1]])
        tail = addrs[-1][0]
        if body:
            msg = 'Unable to connect to port {0} on {1} or {2}'
        else:
            msg = 'Unable to connect to port {0} on {2}'
        super().__init__(None, msg.format(addrs[0][1], body, tail))
        self.errors = errors

    def __reduce__(self):
        return (self.__class__, (self.errors,))