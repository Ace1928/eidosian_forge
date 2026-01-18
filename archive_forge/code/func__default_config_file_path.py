import collections
import json
import os
from tensorflow.python.debug.cli import debugger_cli_common
from tensorflow.python.platform import gfile
def _default_config_file_path(self):
    return os.path.join(os.path.expanduser('~'), self._CONFIG_FILE_NAME)