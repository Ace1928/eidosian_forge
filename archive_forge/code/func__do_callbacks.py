importing any other Kivy modules. Ideally, this means setting them right at
from collections import OrderedDict
from os import environ
from os.path import exists
from weakref import ref
from kivy import kivy_config_fn
from kivy.compat import PY2, string_types
from kivy.logger import Logger, logger_config_update
from kivy.utils import platform
def _do_callbacks(self, section, key, value):
    for callback, csection, ckey in self._callbacks:
        if csection is not None and csection != section:
            continue
        elif ckey is not None and ckey != key:
            continue
        callback(section, key, value)