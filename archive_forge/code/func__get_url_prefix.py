from .. import urlutils
from ..transport import FileExists
from . import decorator
@classmethod
def _get_url_prefix(self):
    """FakeNFS transports are identified by 'brokenrename+'"""
    return 'brokenrename+'