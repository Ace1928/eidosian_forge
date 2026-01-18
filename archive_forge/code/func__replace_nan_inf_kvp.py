import sys
import numpy as np
from pymysql import connect
from pymysql.err import ProgrammingError
from copy import deepcopy
from ase.db.sqlite import SQLite3Database
from ase.db.sqlite import init_statements
from ase.db.sqlite import VERSION
from ase.db.postgresql import remove_nan_and_inf, insert_nan_and_inf
import ase.io.jsonio
import json
def _replace_nan_inf_kvp(self, values):
    for item in values:
        if not np.isfinite(item[1]):
            item[1] = sys.float_info.max / 2
    return values