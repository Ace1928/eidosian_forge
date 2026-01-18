from __future__ import absolute_import, print_function, division
import datetime
import os
import json
import time
import pytest
from petl.compat import text_type
from petl.io.gsheet import fromgsheet, togsheet, appendgsheet
from petl.test.helpers import ieq, get_env_vars_named
def _get_gspread_client():
    credentials = _get_env_credentials()
    try:
        if credentials is None:
            gspread_client = gspread.service_account()
        else:
            gspread_client = gspread.service_account_from_dict(credentials)
    except gspread.exceptions.APIError as ex:
        pytest.skip('SKIPPED. to/from gspread authentication error: %s' % ex)
        return None
    return gspread_client