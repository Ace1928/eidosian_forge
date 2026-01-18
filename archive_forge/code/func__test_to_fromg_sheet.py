from __future__ import absolute_import, print_function, division
import datetime
import os
import json
import time
import pytest
from petl.compat import text_type
from petl.io.gsheet import fromgsheet, togsheet, appendgsheet
from petl.test.helpers import ieq, get_env_vars_named
def _test_to_fromg_sheet(table, sheetname, cell_range, expected):
    filename, gspread_client, emails = _get_gspread_test_params()
    spread_id = togsheet(table, gspread_client, filename, worksheet=sheetname, share_emails=emails)
    try:
        result = fromgsheet(gspread_client, filename, worksheet=sheetname, cell_range=cell_range)
        ieq(expected, result)
    finally:
        gspread_client.del_spreadsheet(spread_id)