import os.path
from pyomo.dataportal import TableData
from pyomo.dataportal.factory import DataManagerFactory
from pyomo.common.errors import ApplicationError
from pyomo.common.dependencies import attempt_import, importlib, pyutilib
def _attempt_open_excel():
    if _attempt_open_excel.result is None:
        from pyutilib.excel.spreadsheet_win32com import ExcelSpreadsheet_win32com
        try:
            tmp = ExcelSpreadsheet_win32com()
            tmp._excel_dispatch()
            tmp._excel_quit()
            _attempt_open_excel.result = True
        except:
            _attempt_open_excel.result = False
    return _attempt_open_excel.result