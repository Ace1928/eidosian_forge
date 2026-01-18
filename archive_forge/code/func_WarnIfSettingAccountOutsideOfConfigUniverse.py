from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.util import exceptions as api_lib_util_exceptions
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.projects import util as command_lib_util
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.credentials import creds as c_creds
from googlecloudsdk.core.credentials import store as c_store
def WarnIfSettingAccountOutsideOfConfigUniverse(account: str, account_universe_domain: str) -> bool:
    """Warn if setting an account belonging to a different universe_domain.

  This warning should only be displayed if the user sets their active account
  to an existing credentialed account which does not match the config
  universe_domain. If the user sets their active account to an uncredentialed
  account, there is no way to determine what universe the account belongs to so
  we do not warn in that case.

  Args:
    account: The account to set [core/account] property to.
    account_universe_domain: The respective account universe domain.

  Returns:
   (Boolean) True if the account is outside of the configuration universe_domain
   and warning is logged. False otherwise.
  """
    config_universe_domain = properties.VALUES.core.universe_domain.Get()
    if account_universe_domain and account_universe_domain != config_universe_domain:
        log.warning(f'This account [{account}] is from the universe domain [{account_universe_domain}] which does not match the current [core/universe_domain] property [{config_universe_domain}]. Update them to match or create a new gcloud configuration for this universe domain using `gcloud config configurations create` with the `--universe-domain` flag or switch to a configuration associated with [{account_universe_domain}].')
        return True
    return False