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
class PromptRecordBase(six.with_metaclass(abc.ABCMeta, object)):
    """Base class to cache prompting results.

  Attributes:
    _cache_file_path: cache file path.
    dirty: bool, True if record in the cache file should be updated. Otherwise,
      False.
    last_prompt_time: Last time user was prompted.
  """

    def __init__(self, cache_file_path=None):
        self._cache_file_path = cache_file_path
        self._dirty = False

    def CacheFileExists(self):
        return os.path.isfile(self._cache_file_path)

    def SavePromptRecordToFile(self):
        """Serializes data to the cache file."""
        if not self._dirty:
            return
        with file_utils.FileWriter(self._cache_file_path) as f:
            yaml.dump(self._ToDictionary(), stream=f)
        self._dirty = False

    @abc.abstractmethod
    def _ToDictionary(self):
        pass

    @abc.abstractmethod
    def ReadPromptRecordFromFile(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.SavePromptRecordToFile()

    @property
    def dirty(self):
        return self._dirty

    @property
    def last_prompt_time(self):
        return self._last_prompt_time

    @last_prompt_time.setter
    def last_prompt_time(self, value):
        self._last_prompt_time = value
        self._dirty = True