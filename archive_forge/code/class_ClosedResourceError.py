from trio._util import NoPublicConstructor, final
class ClosedResourceError(Exception):
    """Raised when attempting to use a resource after it has been closed.

    Note that "closed" here means that *your* code closed the resource,
    generally by calling a method with a name like ``close`` or ``aclose``, or
    by exiting a context manager. If a problem arises elsewhere – for example,
    because of a network failure, or because a remote peer closed their end of
    a connection – then that should be indicated by a different exception
    class, like :exc:`BrokenResourceError` or an :exc:`OSError` subclass.

    """