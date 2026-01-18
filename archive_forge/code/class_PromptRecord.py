from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import time
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files as file_utils
from googlecloudsdk.core.util import prompt_helper
class PromptRecord(prompt_helper.PromptRecordBase):
    """The survey prompt record.

  Attributes:
    _cache_file_path: cache file path.
    last_answer_survey_time: the time user most recently answered the survey
      (epoch time).
    last_prompt_time: the time when user is most recently prompted (epoch time).
    dirty: bool, True if record in the cache file should be updated. Otherwise,
      False.
  """

    def __init__(self):
        super(PromptRecord, self).__init__(cache_file_path=config.Paths().survey_prompting_cache_path)
        self._last_prompt_time, self._last_answer_survey_time = self.ReadPromptRecordFromFile()

    def ReadPromptRecordFromFile(self):
        """Loads the prompt record from the cache file.

    Returns:
       Two-value tuple (last_prompt_time, last_answer_survey_time)
    """
        if not self.CacheFileExists():
            return (None, None)
        try:
            with file_utils.FileReader(self._cache_file_path) as f:
                data = yaml.load(f)
            return (data.get('last_prompt_time', None), data.get('last_answer_survey_time', None))
        except Exception:
            log.debug('Failed to parse survey prompt cache. Using empty cache instead.')
            return (None, None)

    def _ToDictionary(self):
        res = {}
        if self._last_prompt_time is not None:
            res['last_prompt_time'] = self._last_prompt_time
        if self._last_answer_survey_time is not None:
            res['last_answer_survey_time'] = self._last_answer_survey_time
        return res

    @property
    def last_answer_survey_time(self):
        return self._last_answer_survey_time

    @last_answer_survey_time.setter
    def last_answer_survey_time(self, value):
        self._last_answer_survey_time = value
        self._dirty = True