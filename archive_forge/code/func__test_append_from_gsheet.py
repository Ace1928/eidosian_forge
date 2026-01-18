from __future__ import absolute_import, print_function, division
import datetime
import os
import json
import time
import pytest
from petl.compat import text_type
from petl.io.gsheet import fromgsheet, togsheet, appendgsheet
from petl.test.helpers import ieq, get_env_vars_named
def _test_append_from_gsheet(table_list, expected, sheetname=None):
    filename, gspread_client, emails = _get_gspread_test_params()
    table1 = table_list[0]
    other_tables = table_list[1:]
    spread_id = togsheet(table1, gspread_client, filename, worksheet=sheetname, share_emails=emails)
    try:
        for tableN in other_tables:
            appendgsheet(tableN, gspread_client, spread_id, worksheet=sheetname, open_by_key=True)
        result = fromgsheet(gspread_client, spread_id, worksheet=sheetname, open_by_key=True)
        ieq(expected, result)
    finally:
        gspread_client.del_spreadsheet(spread_id)