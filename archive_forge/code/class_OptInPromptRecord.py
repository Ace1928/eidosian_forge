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
class OptInPromptRecord(PromptRecordBase):
    """Opt-in data usage record."""

    def __init__(self):
        super(OptInPromptRecord, self).__init__(cache_file_path=config.Paths().opt_in_prompting_cache_path)
        self._last_prompt_time = self.ReadPromptRecordFromFile()

    def _ToDictionary(self):
        res = {}
        if self._last_prompt_time:
            res['last_prompt_time'] = self._last_prompt_time
        return res

    def ReadPromptRecordFromFile(self):
        if not self.CacheFileExists():
            return None
        try:
            with file_utils.FileReader(self._cache_file_path) as f:
                data = yaml.load(f)
            return data.get('last_prompt_time', None)
        except Exception:
            log.debug('Failed to parse opt-in prompt cache. Using empty cache instead.')
            return None