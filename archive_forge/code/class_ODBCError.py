import os.path
import re
import sys
import shutil
from decimal import Decimal
from pyomo.common.dependencies import attempt_import
from pyomo.dataportal import TableData
from pyomo.dataportal.factory import DataManagerFactory
class ODBCError(Exception):

    def __init__(self, value):
        self.parameter = value

    def __repr__(self):
        return repr(self.parameter)