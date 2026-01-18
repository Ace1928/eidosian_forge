from __future__ import absolute_import, print_function, division
import datetime
import os
import json
import time
import pytest
from petl.compat import text_type
from petl.io.gsheet import fromgsheet, togsheet, appendgsheet
from petl.test.helpers import ieq, get_env_vars_named
def _get_testcase_for_append():
    table_list = [TEST_TABLE[:], TEST_TABLE[:]]
    expected = TEST_TABLE[:] + TEST_TABLE[1:]
    return (table_list, expected)