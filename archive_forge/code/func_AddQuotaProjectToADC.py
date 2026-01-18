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
def AddQuotaProjectToADC(quota_project):
    """Adds the quota project to the existing ADC file.

  Quota project is only added to ADC when the credentials have the
  "serviceusage.services.use" permission on the project.

  Args:
    quota_project: str, The project id of a valid GCP project to add to ADC.

  Raises:
    MissingPermissionOnQuotaProjectError: If the credentials do not have the
      "serviceusage.services.use" permission.
  """
    AssertADCExists()
    if not ADCIsUserAccount():
        raise c_exc.BadFileException('The application default credentials are not user credentials, quota project cannot be added.')
    credentials, _ = c_creds.GetGoogleAuthDefault().load_credentials_from_file(config.ADCFilePath())
    previous_quota_project = credentials.quota_project_id
    adc_path = c_creds.ADC(credentials).DumpExtendedADCToFile(quota_project=quota_project)
    try:
        if not AdcHasGivenPermissionOnProject(quota_project, permissions=[SERVICEUSAGE_PERMISSION]):
            raise MissingPermissionOnQuotaProjectError('Cannot add the project "{}" to application default credentials (ADC) as a quota project because the account in ADC does not have the "{}" permission on this project.'.format(quota_project, SERVICEUSAGE_PERMISSION))
    except Exception:
        c_creds.ADC(credentials).DumpExtendedADCToFile(quota_project=previous_quota_project)
        raise
    LogADCIsWritten(adc_path)
    LogQuotaProjectAdded(quota_project)