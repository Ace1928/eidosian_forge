from ..errors import NoSmartMedium, TransportNotPossible
from ..transport import decorator
class ReadonlyTransportDecorator(decorator.TransportDecorator):
    """A decorator that can convert any transport to be readonly.

    This is requested via the 'readonly+' prefix to get_transport().
    """

    def append_file(self, relpath, f, mode=None):
        """See Transport.append_file()."""
        raise TransportNotPossible('readonly transport')

    def append_bytes(self, relpath, bytes, mode=None):
        """See Transport.append_bytes()."""
        raise TransportNotPossible('readonly transport')

    @classmethod
    def _get_url_prefix(self):
        """Readonly transport decorators are invoked via 'readonly+'"""
        return 'readonly+'

    def rename(self, rel_from, rel_to):
        """See Transport.rename."""
        raise TransportNotPossible('readonly transport')

    def delete(self, relpath):
        """See Transport.delete()."""
        raise TransportNotPossible('readonly transport')

    def delete_tree(self, relpath):
        """See Transport.delete_tree()."""
        raise TransportNotPossible('readonly transport')

    def put_file(self, relpath, f, mode=None):
        """See Transport.put_file()."""
        raise TransportNotPossible('readonly transport')

    def put_bytes(self, relpath: str, raw_bytes: bytes, mode=None):
        """See Transport.put_bytes()."""
        raise TransportNotPossible('readonly transport')

    def mkdir(self, relpath, mode=None):
        """See Transport.mkdir()."""
        raise TransportNotPossible('readonly transport')

    def open_write_stream(self, relpath, mode=None):
        """See Transport.open_write_stream()."""
        raise TransportNotPossible('readonly transport')

    def is_readonly(self):
        """See Transport.is_readonly."""
        return True

    def rmdir(self, relpath):
        """See Transport.rmdir."""
        raise TransportNotPossible('readonly transport')

    def lock_write(self, relpath):
        """See Transport.lock_write."""
        raise TransportNotPossible('readonly transport')

    def get_smart_client(self):
        raise NoSmartMedium(self)

    def get_smart_medium(self):
        raise NoSmartMedium(self)