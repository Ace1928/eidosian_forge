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
def LogADCIsWritten(adc_path):
    """Prints the confirmation when ADC file was successfully written."""
    real_path = adc_path
    if platforms.OperatingSystem.Current() == platforms.OperatingSystem.WINDOWS:
        real_path = GetAdcRealPath(adc_path)
    log.status.Print('\nCredentials saved to file: [{}]'.format(real_path))
    log.status.Print('\nThese credentials will be used by any library that requests Application Default Credentials (ADC).')
    if real_path != adc_path:
        log.warning('You may be running gcloud with a python interpreter installed from Microsoft Store which is not supported by this command. Run `gcloud topic startup` for instructions to select a different python interpreter. Otherwise, you have to set the environment variable `GOOGLE_APPLICATION_CREDENTIALS` to the file path `{}`. See https://cloud.google.com/docs/authentication/getting-started#setting_the_environment_variable for more information.'.format(real_path))