import os.path
import re
import sys
import shutil
from decimal import Decimal
from pyomo.common.dependencies import attempt_import
from pyomo.dataportal import TableData
from pyomo.dataportal.factory import DataManagerFactory

        Parse a string for ODBC sections. The parsing algorithm proceeds
        roughly as follows:

        1. Split the string on newline ('\n') characters.
        2. Remove lines consisting purely of whitespace.
        3. Iterate over lines, storing all key-value pair lines in a dictionary.
        4. When reaching a new section header (denoted by '[str]'), store the old
           key-value pairs under the old section name. Continue from step 3.
        5. On reaching end of data, store the last section and return a mapping
           from section names to dictionaries of key-value pairs in those sections.
        