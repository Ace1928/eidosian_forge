from .. import errors
from ..transport import decorator
class NoSmartTransportDecorator(decorator.TransportDecorator):
    """A decorator for transports that disables get_smart_medium."""

    @classmethod
    def _get_url_prefix(self):
        return 'nosmart+'

    def get_smart_medium(self):
        raise errors.NoSmartMedium(self)