import os.path
from pyomo.dataportal import TableData
from pyomo.dataportal.factory import DataManagerFactory
from pyomo.common.errors import ApplicationError
from pyomo.common.dependencies import attempt_import, importlib, pyutilib
def _spreadsheet_importer():
    pyutilib.component
    return importlib.import_module('pyutilib.excel.spreadsheet')