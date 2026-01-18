from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import datetime
import json
import os
import textwrap
import time
from typing import Optional
import dateutil
from googlecloudsdk.api_lib.auth import util as auth_util
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import transport
from googlecloudsdk.core.configurations import named_configs
from googlecloudsdk.core.credentials import creds as c_creds
from googlecloudsdk.core.credentials import exceptions as creds_exceptions
from googlecloudsdk.core.credentials import gce as c_gce
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
from oauth2client import client
from oauth2client import crypt
from oauth2client import service_account
from oauth2client.contrib import gce as oauth2client_gce
from oauth2client.contrib import reauth_errors
import six
from six.moves import urllib
def _HandleGceUniverseDomain(mds_universe_domain, account):
    """Handles the universe domain from GCE metadata.

  If core/universe_domain property is not explicitly set, set it with the MDS
  universe_domain, but not persist it so it's only used in the current command
  invocation.
  If core/universe_domain property is explicitly set, but it's different from
  the MDS universe_domain, prompt the user to update and persist the
  core/universe_domain property. If the user chooses not to update, an error
  will be raised to avoid sending GCE credentials to a wrong universe domain.

  Args:
    mds_universe_domain: string, The universe domain from metadata server.
    account: string, The account.
  """
    universe_domain_property = properties.VALUES.core.universe_domain
    if not universe_domain_property.IsExplicitlySet():
        universe_domain_property.Set(mds_universe_domain)
        return
    auth_util.HandleUniverseDomainConflict(mds_universe_domain, account)
    if universe_domain_property.Get() != mds_universe_domain:
        raise c_creds.InvalidCredentialsError('Your credentials are from "%(universe_from_mds)s", but your [core/universe_domain] property is set to "%(universe_from_property)s". Update your active account to an account from "%(universe_from_property)s" or update the [core/universe_domain] property to "%(universe_from_mds)s".' % {'universe_from_mds': mds_universe_domain, 'universe_from_property': universe_domain_property.Get()})