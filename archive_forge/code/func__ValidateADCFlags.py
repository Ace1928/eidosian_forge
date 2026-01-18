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
def _ValidateADCFlags(update_adc, add_quota_project_to_adc):
    if not update_adc and add_quota_project_to_adc:
        raise calliope_exceptions.InvalidArgumentException('--add-quota-project-to-adc', '--add-quota-project-to-adc cannot be specified without specifying --update-adc.')