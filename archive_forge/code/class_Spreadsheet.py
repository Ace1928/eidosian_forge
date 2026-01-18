from itertools import zip_longest
import json
import os
from os.path import abspath, dirname, join
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.common.tee import capture_output
from pyomo.dataportal.factory import DataManagerFactory
from pyomo.environ import (
class Spreadsheet(LoadTests):

    def filename(self, tname):
        if tname == 'Z':
            return os.path.abspath(tutorial_dir + os.sep + self._filename) + ' range=' + tname + 'param'
        else:
            return os.path.abspath(tutorial_dir + os.sep + self._filename) + ' range=' + tname + 'table'