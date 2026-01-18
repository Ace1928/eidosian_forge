import atexit
import datetime
import re
import sqlite3
import threading
from pathlib import Path
from decorator import decorator
from traitlets import (
from traitlets.config.configurable import LoggingConfigurable
from IPython.paths import locate_profile
from IPython.utils.decorators import undoc
class HistoryAccessorBase(LoggingConfigurable):
    """An abstract class for History Accessors """

    def get_tail(self, n=10, raw=True, output=False, include_latest=False):
        raise NotImplementedError

    def search(self, pattern='*', raw=True, search_raw=True, output=False, n=None, unique=False):
        raise NotImplementedError

    def get_range(self, session, start=1, stop=None, raw=True, output=False):
        raise NotImplementedError

    def get_range_by_str(self, rangestr, raw=True, output=False):
        raise NotImplementedError