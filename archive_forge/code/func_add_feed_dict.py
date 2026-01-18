import logging
import os
import time
from ray.util.debug import log_once
from ray.rllib.utils.framework import try_import_tf
def add_feed_dict(self, feed_dict):
    assert not self._executed
    for k in feed_dict:
        if k in self.feed_dict:
            raise ValueError('Key added twice: {}'.format(k))
    self.feed_dict.update(feed_dict)