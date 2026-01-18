import os
from traitlets import (
from jupyter_core.paths import jupyter_path
from jupyter_server.transutils import _i18n
from jupyter_server.utils import url_path_join
@observe('mathjax_url')
def _update_mathjax_url(self, change):
    new = change['new']
    if new and (not self.enable_mathjax):
        self.mathjax_url = u''
    else:
        self.log.info(_i18n('Using MathJax: %s'), new)