import os
import sys
import tempfile
from traitlets import default
from .html import HTMLExporter
def _run_pyqtwebengine(self, html):
    ext = '.html'
    temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    filename = f'{temp_file.name[:-len(ext)]}.{self.format}'
    with temp_file:
        temp_file.write(html.encode('utf-8'))
    try:
        QtScreenshot = self._check_launch_reqs()
        s = QtScreenshot()
        s.capture(f'file://{temp_file.name}', filename, self.paginate)
    finally:
        os.unlink(temp_file.name)
    return s.data