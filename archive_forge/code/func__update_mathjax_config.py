import os
from traitlets import (
from jupyter_core.paths import jupyter_path
from jupyter_server.transutils import _i18n
from jupyter_server.utils import url_path_join
@observe('mathjax_config')
def _update_mathjax_config(self, change):
    self.log.info(_i18n('Using MathJax configuration file: %s'), change['new'])