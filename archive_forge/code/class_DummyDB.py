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
@undoc
class DummyDB(object):
    """Dummy DB that will act as a black hole for history.

    Only used in the absence of sqlite"""

    def execute(*args, **kwargs):
        return []

    def commit(self, *args, **kwargs):
        pass

    def __enter__(self, *args, **kwargs):
        pass

    def __exit__(self, *args, **kwargs):
        pass