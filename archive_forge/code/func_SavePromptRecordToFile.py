from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import os
import time
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files as file_utils
import six
def SavePromptRecordToFile(self):
    """Serializes data to the cache file."""
    if not self._dirty:
        return
    with file_utils.FileWriter(self._cache_file_path) as f:
        yaml.dump(self._ToDictionary(), stream=f)
    self._dirty = False