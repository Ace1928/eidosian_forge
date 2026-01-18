from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
import textwrap
from google.auth import jwt
from googlecloudsdk.api_lib.auth import exceptions as auth_exceptions
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.api_lib.iamcredentials import util as impersonation_util
from googlecloudsdk.calliope import exceptions as c_exc
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.credentials import creds as c_creds
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
from oauth2client import client
from oauth2client import service_account
from oauth2client.contrib import gce as oauth2client_gce
def DumpADCOptionalQuotaProject(credentials):
    """Dumps the given credentials to ADC file with an optional quota project.

  Loads quota project from gcloud's context and writes it to application default
  credentials file if the credentials has the "serviceusage.services.use"
  permission on the quota project..

  Args:
     credentials: a credentials from oauth2client or google-auth libraries, the
       credentials to dump.
  """
    adc_path = c_creds.ADC(credentials).DumpADCToFile()
    LogADCIsWritten(adc_path)
    quota_project = c_creds.GetQuotaProject(credentials, force_resource_quota=True)
    if not quota_project:
        LogQuotaProjectNotFound()
    elif AdcHasGivenPermissionOnProject(quota_project, permissions=[SERVICEUSAGE_PERMISSION]):
        c_creds.ADC(credentials).DumpExtendedADCToFile(quota_project=quota_project)
        LogQuotaProjectAdded(quota_project)
    else:
        LogMissingPermissionOnQuotaProject(quota_project)