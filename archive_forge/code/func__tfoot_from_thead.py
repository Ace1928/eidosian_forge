import json
import logging
import re
import uuid
import warnings
from base64 import b64encode
from pathlib import Path
import numpy as np
import pandas as pd
from .utils import UNPKG_DT_BUNDLE_CSS, UNPKG_DT_BUNDLE_URL
from .version import __version__ as itables_version
from IPython.display import HTML, display
import itables.options as opt
from .datatables_format import datatables_rows
from .downsample import downsample
from .utils import read_package_file
def _tfoot_from_thead(thead):
    header_rows = thead.split('</tr>')
    last_row = header_rows[-1]
    assert not last_row.strip(), last_row
    header_rows = header_rows[:-1]
    return ''.join((row + '</tr>' for row in header_rows[::-1] if '<tr' in row)) + '\n'