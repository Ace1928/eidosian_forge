import os.path
import re
import sys
import shutil
from decimal import Decimal
from pyomo.common.dependencies import attempt_import
from pyomo.dataportal import TableData
from pyomo.dataportal.factory import DataManagerFactory
class db_Table(TableData):

    def __init__(self):
        TableData.__init__(self)
        self.using = None

    def open(self):
        if self.filename is None:
            raise IOError('No data source name was specified')
        if self.filename[0] == '"':
            self.filename = self.filename[1:-1]
        if not self.using is None:
            self.options.using = self.using
        self.db = None
        if self._data is not None:
            self.db = self._data
        else:
            try:
                self.db = self.connect(self.filename, self.options)
            except Exception:
                raise

    def read(self):
        if self.db is None:
            return
        cursor = self.db.cursor()
        tmp = []
        if self.options.query is None:
            if self.options.table is None:
                raise IOError("Must specify 'query' or 'table' option!")
            self.options.query = 'SELECT * FROM %s' % self.options.table
        elif self.options.query[0] in ("'", '"'):
            self.options.query = self.options.query[1:-1]
        try:
            cursor.execute(self.options.query)
            rows = cursor.fetchall()
            for col in cursor.description:
                tmp.append(col[0])
            tmp = [tmp]
        except sqlite3.OperationalError:
            import logging
            logging.getLogger('pyomo.core').error('Fatal error reading from an external ODBC data source.\n\nThis error was generated outside Pyomo by the Python connector to the\nexternal data source:\n\n    %s\n\nfor the query:\n\n    %s\n\nIt is possible that you have an error in your external data file,\nthe ODBC connector for this data source is not correctly installed,\nor that there is a bug in the ODBC connector.\n' % (self.filename, self.options.query))
            raise
        for row in rows:
            ttmp = []
            for data in list(row):
                if isinstance(data, Decimal):
                    ttmp.append(float(data))
                elif data is None:
                    ttmp.append('.')
                elif isinstance(data, str):
                    nulidx = data.find('\x00')
                    if nulidx > -1:
                        data = data[:nulidx]
                    ttmp.append(data)
                else:
                    ttmp.append(data)
            tmp.append(ttmp)
        if type(tmp) in (int, float):
            if not self.options.param is None:
                self._info = ['param', self.options.param.local_name, ':=', tmp]
            elif len(self.options.symbol_map) == 1:
                self._info = ['param', self.options.symbol_map[self.options.symbol_map.keys()[0]], ':=', tmp]
            else:
                raise IOError('Data looks like a scalar parameter, but multiple parameter names have been specified: %s' % str(self.options.symbol_map))
        elif len(tmp) == 0:
            raise IOError("Empty range '%s'" % self.options.range)
        else:
            self._set_data(tmp[0], tmp[1:])

    def close(self):
        if self._data is None and (not self.db is None):
            del self.db

    def connect(self, connection, options, kwds={}):
        try:
            mod = __import__(options.using)
            args = [connection]
            if not options.user is None:
                args.append(options.user)
            if not options.password is None:
                args.append(options.password)
            if not options.database is None:
                args.append(options.database)
            return mod.connect(*args, **kwds)
        except ImportError:
            return None