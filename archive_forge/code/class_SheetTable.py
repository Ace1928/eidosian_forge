import os.path
from pyomo.dataportal import TableData
from pyomo.dataportal.factory import DataManagerFactory
from pyomo.common.errors import ApplicationError
from pyomo.common.dependencies import attempt_import, importlib, pyutilib
class SheetTable(TableData):

    def __init__(self, ctype=None):
        TableData.__init__(self)
        self.ctype = ctype

    def open(self):
        if self.filename is None:
            raise IOError('No filename specified')
        if not os.path.exists(self.filename):
            raise IOError("Cannot find file '%s'" % self.filename)
        self.sheet = None
        if self._data is not None:
            self.sheet = self._data
        else:
            try:
                self.sheet = spreadsheet.ExcelSpreadsheet(self.filename, ctype=self.ctype)
            except ApplicationError:
                raise

    def read(self):
        if self.sheet is None:
            return
        tmp = self.sheet.get_range(self.options.range, raw=True)
        if type(tmp) is float or type(tmp) is int:
            if not self.options.param is None:
                self._info = ['param'] + list(self.options.param) + [':=', tmp]
            elif len(self.options.symbol_map) == 1:
                self._info = ['param', self.options.symbol_map[self.options.symbol_map.keys()[0]], ':=', tmp]
            else:
                raise IOError('Data looks like a parameter, but multiple parameter names have been specified: %s' % str(self.options.symbol_map))
        elif len(tmp) == 0:
            raise IOError("Empty range '%s'" % self.options.range)
        else:
            if type(tmp[1]) in (list, tuple):
                tmp_ = tmp[1:]
            else:
                tmp_ = [[x] for x in tmp[1:]]
            self._set_data(tmp[0], tmp_)

    def close(self):
        if self._data is None and (not self.sheet is None):
            del self.sheet