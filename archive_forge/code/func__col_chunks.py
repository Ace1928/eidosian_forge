import os
import re
import string
import textwrap
import warnings
from string import Formatter
from pathlib import Path
from typing import List, Dict, Tuple, Optional, cast
def _col_chunks(l, max_rows, row_first=False):
    """Yield successive max_rows-sized column chunks from l."""
    if row_first:
        ncols = len(l) // max_rows + (len(l) % max_rows > 0)
        for i in range(ncols):
            yield [l[j] for j in range(i, len(l), ncols)]
    else:
        for i in range(0, len(l), max_rows):
            yield l[i:i + max_rows]