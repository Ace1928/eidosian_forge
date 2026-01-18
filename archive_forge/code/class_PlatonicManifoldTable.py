from __future__ import print_function
import sys, sqlite3, re, os, random
from .sqlite_files import __path__ as manifolds_paths
class PlatonicManifoldTable(ManifoldTable):
    """
        Iterator for platonic hyperbolic manifolds.
        """

    def __init__(self, table='', db_path=platonic_database_path, **filter_args):
        ManifoldTable.__init__(self, table=table, db_path=db_path, **filter_args)

    def _configure(self, **kwargs):
        ManifoldTable._configure(self, **kwargs)
        conditions = []
        if 'solids' in kwargs:
            N = int(kwargs['solids'])
            conditions.append('solids = %d' % N)
        if self._filter:
            if len(conditions) > 0:
                self._filter += ' and ' + ' and '.join(conditions)
        else:
            self._filter = ' and '.join(conditions)