import re
from . import decorator
class FakeVFATTransportDecorator(decorator.TransportDecorator):
    """A decorator that can convert any transport to be readonly.

    This is requested via the 'vfat+' prefix to get_transport().

    This is intended only for use in testing and doesn't implement every
    method very well yet.

    This transport is typically layered on a local or memory transport
    which actually stored the files.
    """

    def _can_roundtrip_unix_modebits(self):
        """See Transport._can_roundtrip_unix_modebits()."""
        return False

    @classmethod
    def _get_url_prefix(self):
        """Readonly transport decorators are invoked via 'vfat+'"""
        return 'vfat+'

    def _squash_name(self, name):
        """Return vfat-squashed filename.

        The name is returned as it will be stored on disk.  This raises an
        error if there are invalid characters in the name.
        """
        if re.search('[?*:;<>]', name):
            raise ValueError('illegal characters for VFAT filename: %r' % name)
        return name.lower()

    def get(self, relpath):
        return self._decorated.get(self._squash_name(relpath))

    def mkdir(self, relpath, mode=None):
        return self._decorated.mkdir(self._squash_name(relpath), 493)

    def has(self, relpath):
        return self._decorated.has(self._squash_name(relpath))

    def _readv(self, relpath, offsets):
        return self._decorated.readv(self._squash_name(relpath), offsets)

    def put_file(self, relpath, f, mode=None):
        return self._decorated.put_file(self._squash_name(relpath), f, mode)