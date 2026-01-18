import os
from traitlets import (
from jupyter_core.paths import jupyter_path
from jupyter_server.transutils import _i18n
from jupyter_server.utils import url_path_join
@default('static_custom_path')
def _default_static_custom_path(self):
    return [os.path.join(self.config_dir, 'custom')]