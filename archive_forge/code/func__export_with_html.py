import os
import sys
import time
import traceback
import param
from IPython.display import Javascript, display
from nbconvert import HTMLExporter, NotebookExporter
from nbconvert.preprocessors.clearoutput import ClearOutputPreprocessor
from nbformat import reader
from ..core.io import FileArchive, Pickler
from ..plotting.renderer import HTML_TAGS, MIME_TYPES
from .preprocessors import Substitute
def _export_with_html(self):
    """Computes substitutions before using nbconvert with preprocessors"""
    self.export_success = False
    try:
        tstamp = time.strftime(self.timestamp_format, self._timestamp)
        substitutions = {}
        for (basename, ext), entry in self._files.items():
            _, info = entry
            html_key = self._replacements.get((basename, ext), None)
            if html_key is None:
                continue
            filename = self._format(basename, {'timestamp': tstamp, 'notebook': self.notebook_name})
            fpath = filename + (f'.{ext}' if ext else '')
            info = {'src': fpath, 'mime_type': info['mime_type']}
            if 'mime_type' not in info:
                pass
            elif info['mime_type'] not in self._tags:
                pass
            else:
                basename, ext = os.path.splitext(fpath)
                truncated = self._truncate_name(basename, ext[1:])
                link_html = self._format(self._tags[info['mime_type']], {'src': truncated, 'mime_type': info['mime_type'], 'css': ''})
                substitutions[html_key] = (link_html, truncated)
        node = self._get_notebook_node()
        html = self._generate_html(node, substitutions)
        export_filename = self.snapshot_name
        info = {'file-ext': 'html', 'mime_type': 'text/html', 'notebook': self.notebook_name}
        super().add(filename=export_filename, data=html, info=info)
        cleared = self._clear_notebook(node)
        info = {'file-ext': 'ipynb', 'mime_type': 'text/json', 'notebook': self.notebook_name}
        super().add(filename=export_filename, data=cleared, info=info)
        super().export(timestamp=self._timestamp, info={'notebook': self.notebook_name})
    except Exception:
        self.traceback = traceback.format_exc()
    else:
        self.export_success = True