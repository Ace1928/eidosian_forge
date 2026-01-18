from ..api import CacheBackend
from ..api import DefaultSerialization
from ..api import NO_VALUE
class MemoryPickleBackend(DefaultSerialization, MemoryBackend):
    """A backend that uses a plain dictionary, but serializes objects on
    :meth:`.MemoryBackend.set` and deserializes :meth:`.MemoryBackend.get`.

    E.g.::

        from dogpile.cache import make_region

        region = make_region().configure(
            'dogpile.cache.memory_pickle'
        )

    The usage of pickle to serialize cached values allows an object
    as placed in the cache to be a copy of the original given object, so
    that any subsequent changes to the given object aren't reflected
    in the cached value, thus making the backend behave the same way
    as other backends which make use of serialization.

    The serialization is performed via pickle, and incurs the same
    performance hit in doing so as that of other backends; in this way
    the :class:`.MemoryPickleBackend` performance is somewhere in between
    that of the pure :class:`.MemoryBackend` and the remote server oriented
    backends such as that of Memcached or Redis.

    Pickle behavior here is the same as that of the Redis backend, using
    either ``cPickle`` or ``pickle`` and specifying ``HIGHEST_PROTOCOL``
    upon serialize.

    .. versionadded:: 0.5.3

    """