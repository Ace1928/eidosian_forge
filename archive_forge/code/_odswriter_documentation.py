from __future__ import annotations
from collections import defaultdict
import datetime
import json
from typing import (
from pandas.io.excel._base import ExcelWriter
from pandas.io.excel._util import (

        Create freeze panes in the sheet.

        Parameters
        ----------
        sheet_name : str
            Name of the spreadsheet
        freeze_panes : tuple of (int, int)
            Freeze pane location x and y
        