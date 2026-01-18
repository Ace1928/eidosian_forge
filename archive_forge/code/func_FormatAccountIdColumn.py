from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import base64
import collections
import copy
import datetime
import enum
import hashlib
import json
import os
import sqlite3
from google.auth import compute_engine as google_auth_compute_engine
from google.auth import credentials as google_auth_creds
from google.auth import exceptions as google_auth_exceptions
from google.auth import external_account as google_auth_external_account
from google.auth import external_account_authorized_user as google_auth_external_account_authorized_user
from google.auth import impersonated_credentials as google_auth_impersonated
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.credentials import devshell as c_devshell
from googlecloudsdk.core.credentials import exceptions as c_exceptions
from googlecloudsdk.core.credentials import introspect as c_introspect
from googlecloudsdk.core.util import files
from oauth2client import client
from oauth2client import service_account
from oauth2client.contrib import gce as oauth2client_gce
import six
def FormatAccountIdColumn(self):
    """Format the account id column.

    Before we introduce the formatted account id concept, the existing table
    uses the account id value as the key. Therefore we need to load the table
    and replace these account ids with formatted account ids.
    """
    with self._cursor as cur:
        table = cur.Execute('SELECT account_id, value FROM "{}"'.format(_CREDENTIAL_TABLE_NAME)).fetchall()
        for row in table:
            account_id, cred_json = (row[0], row[1])
            if '#' not in account_id:
                creds = FromJsonGoogleAuth(cred_json)
                formatted_account_id = _AccountIdFormatter.GetFormattedAccountId(account_id, creds)
                if account_id != formatted_account_id:
                    cur.Execute('DELETE FROM "{}" WHERE account_id = ?'.format(_CREDENTIAL_TABLE_NAME), (account_id,))
                    cur.Execute('INSERT INTO "{}" (account_id, value) VALUES (?,?)'.format(_CREDENTIAL_TABLE_NAME), (formatted_account_id, cred_json))
        config_store = config.GetConfigStore()
        config_store.Set('cred_token_store_formatted', True)