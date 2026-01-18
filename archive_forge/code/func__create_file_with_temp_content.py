import os
import tempfile
import unittest
from .config_exception import ConfigException
from .incluster_config import (SERVICE_HOST_ENV_NAME, SERVICE_PORT_ENV_NAME,
def _create_file_with_temp_content(self, content=''):
    handler, name = tempfile.mkstemp()
    self._temp_files.append(name)
    os.write(handler, str.encode(content))
    os.close(handler)
    return name