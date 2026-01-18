import os.path
from pyomo.dataportal import TableData
from pyomo.dataportal.factory import DataManagerFactory
from pyomo.common.errors import ApplicationError
from pyomo.common.dependencies import attempt_import, importlib, pyutilib
@DataManagerFactory.register('xls', 'Excel XLS file interface')
class SheetTable_xls(SheetTable):

    def __init__(self):
        if spreadsheet.Interfaces()['win32com'].available and _attempt_open_excel():
            SheetTable.__init__(self, ctype='win32com')
        elif spreadsheet.Interfaces()['xlrd'].available:
            SheetTable.__init__(self, ctype='xlrd')
        else:
            raise RuntimeError('No excel interface is available; install %s' % self.requirements())

    def available(self):
        _inter = spreadsheet.Interfaces()
        return _inter['win32com'].available and _attempt_open_excel() or _inter['xlrd'].available

    def requirements(self):
        return 'win32com or xlrd'