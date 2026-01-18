from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
import importlib
import logging
import os
import sys
import threading
from googlecloudsdk.core.util import encoding
def _update_defaults(self, mapping):
    """Updates the default mappings.

    Args:
      mapping: A dict mapping suffix strings to default values.
    """
    self._lock.acquire()
    try:
        for key, value in mapping.items():
            if key.startswith('__') and key.endswith('__'):
                continue
            self._defaults[key] = value
        if self._initialized:
            self._update_configs()
    finally:
        self._lock.release()