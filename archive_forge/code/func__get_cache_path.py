from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import files
def _get_cache_path():
    config_dir = config.Paths().global_config_dir
    return os.path.join(config_dir, WORKFLOW_CACHE_FILE)