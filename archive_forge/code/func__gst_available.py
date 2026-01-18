from . import ffdec
from .exceptions import DecodeError, NoBackendError
from .version import version as __version__  # noqa
from .base import AudioFile  # noqa
def _gst_available():
    """Determine whether Gstreamer and the Python GObject bindings are
    installed.
    """
    try:
        import gi
    except ImportError:
        return False
    try:
        gi.require_version('Gst', '1.0')
    except (ValueError, AttributeError):
        return False
    try:
        from gi.repository import Gst
    except ImportError:
        return False
    return True