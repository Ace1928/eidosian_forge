import hashlib
import os
from jupyterlab_pygments import JupyterStyle  # type:ignore[import-untyped]
from pygments.style import Style
from traitlets import Type, Unicode, Union
from .base import Preprocessor
def _generate_header(self, resources):
    """
        Fills self.header with lines of CSS extracted from IPython
        and Pygments.
        """
    from pygments.formatters import HtmlFormatter
    header = []
    formatter = HtmlFormatter(style=self.style)
    pygments_css = formatter.get_style_defs(self.highlight_class)
    header.append(pygments_css)
    config_dir = resources['config_dir']
    custom_css_filename = os.path.join(config_dir, 'custom', 'custom.css')
    if os.path.isfile(custom_css_filename):
        if DEFAULT_STATIC_FILES_PATH and self._default_css_hash is None:
            self._default_css_hash = self._hash(os.path.join(DEFAULT_STATIC_FILES_PATH, 'custom', 'custom.css'))
        if self._hash(custom_css_filename) != self._default_css_hash:
            with open(custom_css_filename, encoding='utf-8') as f:
                header.append(f.read())
    return header