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
def WarnIfSettingUniverseDomainOutsideOfConfigAccountUniverse(universe_domain: str) -> bool:
    """Warn if setting a universe domain mismatched to config account domain.

  This warning should only be displayed if the user sets their universe domain
  property to a universe domain not associated with the current credentialed
  account. If the user has their config set to an uncredentialed account, there
  is no way to determine what universe that account belongs to so we do not warn
  in that case.

  Args:
    universe_domain: The universe domain to set [core/universe_domain] property
      to.

  Returns:
    (Boolean) True if the provided universe_domain is outside of the
    configuration universe_domain and warning is logged. False otherwise.
  """
    config_account = properties.VALUES.core.account.Get()
    all_cred_accounts = c_store.AllAccountsWithUniverseDomains()
    cred_universe_domains = []
    for cred_account in all_cred_accounts:
        if cred_account.account == config_account:
            cred_universe_domains.append(cred_account.universe_domain)
    if cred_universe_domains and universe_domain not in cred_universe_domains:
        cred_universe_domain_list = ', '.join(cred_universe_domains)
        log.warning(f'The config account [{config_account}] is available in the following universe domain(s): [{cred_universe_domain_list}], but it is not available in [{universe_domain}] which is specified by the [core/universe_domain] property. Update them to match or create a new gcloud configuration for this universe domain using `gcloud config configurations create` with the `--universe-domain` flag or switch to a configuration associated with [{cred_universe_domain_list}].')
        return True
    return False