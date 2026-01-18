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
def _writeout_input_cache(self, conn):
    with conn:
        for line in self.db_input_cache:
            conn.execute('INSERT INTO history VALUES (?, ?, ?, ?)', (self.session_number,) + line)