import os
import sys
import tempfile
from traitlets import default
from .html import HTMLExporter
def _check_launch_reqs(self):
    if sys.platform.startswith('win') and self.format == 'png':
        msg = 'Exporting to PNG using Qt is currently not supported on Windows.'
        raise RuntimeError(msg)
    from .qt_screenshot import QT_INSTALLED
    if not QT_INSTALLED:
        msg = f'PyQtWebEngine is not installed to support Qt {self.format.upper()} conversion. Please install `nbconvert[qt{self.format}]` to enable.'
        raise RuntimeError(msg)
    from .qt_screenshot import QtScreenshot
    return QtScreenshot