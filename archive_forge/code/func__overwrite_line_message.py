import contextlib
from datetime import datetime
import sys
import time
def _overwrite_line_message(self, message, color_code=_STYLE_GREEN):
    """Overwrite the current line with a stylized message."""
    if not self._verbosity:
        return
    message += '.' * 3
    sys.stdout.write(_STYLE_ERASE_LINE + color_code + message + _STYLE_RESET + '\r')
    sys.stdout.flush()