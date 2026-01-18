from datetime import datetime
from inspect import signature
from io import StringIO
import os
from pathlib import Path
import sys
import numpy as np
import pytest
from pandas.errors import (
from pandas import (
import pandas._testing as tm
from pandas.io.parsers import TextFileReader
from pandas.io.parsers.c_parser_wrapper import CParserWrapper
class MyCParserWrapper(CParserWrapper):

    def _set_noconvert_columns(self):
        if self.usecols_dtype == 'integer':
            self.usecols = list(self.usecols)
            self.usecols.reverse()
        return CParserWrapper._set_noconvert_columns(self)