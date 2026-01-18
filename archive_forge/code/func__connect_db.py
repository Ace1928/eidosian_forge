from __future__ import annotations
import hashlib
import os
import sys
import typing as t
from collections import OrderedDict
from contextlib import contextmanager
from datetime import datetime, timezone
from hmac import HMAC
from pathlib import Path
from base64 import encodebytes
from jupyter_core.application import JupyterApp, base_flags
from traitlets import Any, Bool, Bytes, Callable, Enum, Instance, Integer, Unicode, default, observe
from traitlets.config import LoggingConfigurable, MultipleInstanceError
from . import NO_CONVERT, __version__, read, reads
def _connect_db(self, db_file):
    kwargs: dict[str, t.Any] = {'detect_types': sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES}
    db = None
    try:
        db = sqlite3.connect(db_file, **kwargs)
        self.init_db(db)
    except (sqlite3.DatabaseError, sqlite3.OperationalError):
        if db_file != ':memory:':
            old_db_location = db_file + '.bak'
            if db is not None:
                db.close()
            self.log.warning('The signatures database cannot be opened; maybe it is corrupted or encrypted. You may need to rerun your notebooks to ensure that they are trusted to run Javascript. The old signatures database has been renamed to %s and a new one has been created.', old_db_location)
            try:
                Path(db_file).rename(old_db_location)
                db = sqlite3.connect(db_file, **kwargs)
                self.init_db(db)
            except (sqlite3.DatabaseError, sqlite3.OperationalError, OSError):
                if db is not None:
                    db.close()
                self.log.warning('Failed committing signatures database to disk. You may need to move the database file to a non-networked file system, using config option `NotebookNotary.db_file`. Using in-memory signatures database for the remainder of this session.')
                self.db_file = ':memory:'
                db = sqlite3.connect(':memory:', **kwargs)
                self.init_db(db)
        else:
            raise
    return db