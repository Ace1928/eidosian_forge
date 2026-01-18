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
class JavascriptFunction(str):
    """A class that explicitly states that a string is a Javascript function"""

    def __init__(self, value):
        assert value.lstrip().startswith('function'), "A Javascript function is expected to start with 'function'"