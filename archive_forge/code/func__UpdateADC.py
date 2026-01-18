from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.auth import external_account as auth_external_account
from googlecloudsdk.api_lib.auth import service_account as auth_service_account
from googlecloudsdk.api_lib.auth import util as auth_util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.auth import auth_util as command_auth_util
from googlecloudsdk.command_lib.auth import flags as auth_flags
from googlecloudsdk.command_lib.auth import workforce_login_config as workforce_login_config_util
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.credentials import creds as c_creds
from googlecloudsdk.core.credentials import devshell as c_devshell
from googlecloudsdk.core.credentials import exceptions as creds_exceptions
from googlecloudsdk.core.credentials import gce as c_gce
from googlecloudsdk.core.credentials import store as c_store
def _UpdateADC(creds, add_quota_project_to_adc):
    """Updates the ADC json with the credentials creds."""
    old_adc_json = command_auth_util.GetADCAsJson()
    command_auth_util.WriteGcloudCredentialsToADC(creds, add_quota_project_to_adc)
    new_adc_json = command_auth_util.GetADCAsJson()
    if new_adc_json and new_adc_json != old_adc_json:
        adc_msg = '\nApplication Default Credentials (ADC) were updated.'
        quota_project = command_auth_util.GetQuotaProjectFromADC()
        if quota_project:
            adc_msg = adc_msg + "\n'{}' is added to ADC as the quota project.\nTo just update the quota project in ADC, use $gcloud auth application-default set-quota-project.".format(quota_project)
        log.status.Print(adc_msg)