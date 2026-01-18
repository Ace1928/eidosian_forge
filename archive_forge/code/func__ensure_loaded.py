import pkgutil
import sys
from _pydev_bundle import pydev_log
def _ensure_loaded(self):
    if self.loaded_extensions is None:
        self._load_modules()