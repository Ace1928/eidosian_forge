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
def _writeout_output_cache(self, conn):
    with conn:
        for line in self.db_output_cache:
            conn.execute('INSERT INTO output_history VALUES (?, ?, ?)', (self.session_number,) + line)