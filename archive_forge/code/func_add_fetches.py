import logging
import os
import time
from ray.util.debug import log_once
from ray.rllib.utils.framework import try_import_tf
def add_fetches(self, fetches):
    assert not self._executed
    base_index = len(self.fetches)
    self.fetches.extend(fetches)
    return list(range(base_index, len(self.fetches)))