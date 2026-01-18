import os.path
import re
import sys
import shutil
from decimal import Decimal
from pyomo.common.dependencies import attempt_import
from pyomo.dataportal import TableData
from pyomo.dataportal.factory import DataManagerFactory
@DataManagerFactory.register('pymysql', 'pymysql database interface')
class pymysql_db_Table(db_Table):

    def __init__(self):
        db_Table.__init__(self)
        self.using = 'pymysql'

    def available(self):
        return pymysql_available

    def requirements(self):
        return 'pymysql'