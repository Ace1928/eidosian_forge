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
class _AccountIdFormatter(object):
    """Account ID formatter.

  In this file, when we say "account id" or "account_id", it means principal;
  when we say "formatted_account_id" or "formatted account id", it means:
    - the account_id or principal, if the universe domain is GDU
    - the "account_id#universe_domain" string, otherwise

  In credentials and access token sqlite3 database, the account_id column saves
  the formatted account id.

  This class provides utility functions to handle the formatting.
  """

    @staticmethod
    def GetAccountIdAndUniverseDomain(formatted_account_id):
        """Get account_id/principal and universe domain from formatted account id.

    Args:
      formatted_account_id: str, the formatted account id string.

    Returns:
      tuple: The principal and universe domain tuple.
    """
        splits = formatted_account_id.split('#')
        principal = splits[0]
        if len(splits) == 1:
            universe_domain = properties.VALUES.core.universe_domain.default
        else:
            universe_domain = splits[1]
        return (principal, universe_domain)

    @staticmethod
    def GetFormattedAccountId(account_id, credentials=None):
        """Calculate the formatted account id.

    If the universe_domain is GDU, return the account_id as is; otherwise,
    return "account_id#universe_domain". Here the universe_domain value comes
    from the credentials if it's provided, otherwise it comes from the
    core/universe_domain property.

    Args:
      account_id: str, the account id or principal string.
      credentials: google.auth.credentials.Credentials, The optional credentials
        provided to derive the universe_domain value.

    Returns:
      str: The formatted account id.
    """
        universe_domain_property = properties.VALUES.core.universe_domain
        if credentials and hasattr(credentials, 'universe_domain'):
            universe_domain = credentials.universe_domain
        else:
            universe_domain = universe_domain_property.Get()
        if universe_domain == universe_domain_property.default:
            return account_id
        else:
            return account_id + '#' + universe_domain