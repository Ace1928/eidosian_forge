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
@default('dir_hist')
def _dir_hist_default(self):
    try:
        return [Path.cwd()]
    except OSError:
        return []