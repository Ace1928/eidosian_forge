import os.path
import typing as t
from jupyter_core.paths import jupyter_config_dir, jupyter_config_path
from traitlets import Instance, List, Unicode, default, observe
from traitlets.config import LoggingConfigurable
from jupyter_server.config_manager import BaseJSONConfigManager, recursive_update
@observe('write_config_dir')
def _update_write_config_dir(self, change):
    self.write_config_manager = BaseJSONConfigManager(config_dir=self.write_config_dir)