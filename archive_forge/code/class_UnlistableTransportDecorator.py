from ..transport import Transport, decorator
class UnlistableTransportDecorator(decorator.TransportDecorator):
    """A transport that disables file listing for testing."""

    @classmethod
    def _get_url_prefix(self):
        """Unlistable transports are identified by 'unlistable+'"""
        return 'unlistable+'

    def iter_files_recursive(self):
        Transport.iter_files_recursive(self)

    def listable(self):
        return False

    def list_dir(self, relpath):
        Transport.list_dir(self, relpath)