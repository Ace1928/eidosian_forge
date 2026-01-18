import contextlib
import getpass
import logging
import os
import sqlite3
import tempfile
import warnings
from io import BytesIO
from os.path import join as pjoin
import numpy
from nibabel.optpkg import optional_package
from .nifti1 import Nifti1Header
class _DB:

    def __init__(self, fname=None, verbose=True):
        self.fname = fname or pjoin(tempfile.gettempdir(), f'dft.{getpass.getuser()}.sqlite')
        self.verbose = verbose

    @property
    def session(self):
        """Get sqlite3 Connection

        The connection is created on the first call of this property
        """
        try:
            return self._session
        except AttributeError:
            self._init_db()
            return self._session

    def _init_db(self):
        if self.verbose:
            logger.info('db filename: ' + self.fname)
        self._session = sqlite3.connect(self.fname, isolation_level='EXCLUSIVE')
        with self.readwrite_cursor() as c:
            c.execute("SELECT COUNT(*) FROM sqlite_master WHERE type = 'table'")
            if c.fetchone()[0] == 0:
                logger.debug('create')
                for q in CREATE_QUERIES:
                    c.execute(q)

    def __repr__(self):
        return f'<DFT {self.fname!r}>'

    @contextlib.contextmanager
    def readonly_cursor(self):
        cursor = self.session.cursor()
        try:
            yield cursor
        finally:
            cursor.close()
            self.session.rollback()

    @contextlib.contextmanager
    def readwrite_cursor(self):
        cursor = self.session.cursor()
        try:
            yield cursor
        except Exception:
            self.session.rollback()
            raise
        finally:
            cursor.close()
        self.session.commit()