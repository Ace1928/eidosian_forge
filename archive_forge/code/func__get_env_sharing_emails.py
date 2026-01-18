from __future__ import absolute_import, print_function, division
import datetime
import os
import json
import time
import pytest
from petl.compat import text_type
from petl.io.gsheet import fromgsheet, togsheet, appendgsheet
from petl.test.helpers import ieq, get_env_vars_named
def _get_env_sharing_emails():
    emails = get_env_vars_named('PETL_GSHEET_EMAIL', remove_prefix=False)
    if emails is not None:
        return list(emails.values())
    return []