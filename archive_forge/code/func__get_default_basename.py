import os
from sphinx.util import logging
from oslo_config import generator
def _get_default_basename(config_file):
    return os.path.splitext(os.path.basename(config_file))[0]