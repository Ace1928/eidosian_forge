from __future__ import absolute_import, print_function, division
import datetime
import os
import json
import time
import pytest
from petl.compat import text_type
from petl.io.gsheet import fromgsheet, togsheet, appendgsheet
from petl.test.helpers import ieq, get_env_vars_named
def _get_gspread_credentials():
    json_path = os.getenv('PETL_GCP_JSON_PATH', None)
    if json_path is not None and os.path.exists(json_path):
        return json_path
    json_props = get_env_vars_named('PETL_GCP_CREDS_')
    if json_props is not None:
        return json_props
    user_path = os.path.expanduser('~/.config/gspread/service_account.json')
    if os.path.isfile(user_path) and os.path.exists(user_path):
        return user_path
    return None