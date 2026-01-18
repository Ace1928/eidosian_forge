from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import os
from googlecloudsdk.api_lib import apigee
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.apigee import errors
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
def _SaveCachedDataWithName(data, name):
    """Saves `data` to a named cache file.

  Cache files are saved as hidden YAML files in the gcloud config directory.

  Args:
    data: The data to cache.
    name: The name of the cache file.
  """
    config_dir = config.Paths().global_config_dir
    cache_path = os.path.join(config_dir, '.apigee-cached-' + name)
    files.WriteFileContents(cache_path, yaml.dump(data))