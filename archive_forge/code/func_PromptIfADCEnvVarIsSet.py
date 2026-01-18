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
def PromptIfADCEnvVarIsSet():
    """Warns users if ADC environment variable is set."""
    override_file = config.ADCEnvVariable()
    if override_file:
        message = textwrap.dedent('\n          The environment variable [{envvar}] is set to:\n            [{override_file}]\n          Credentials will still be generated to the default location:\n            [{default_file}]\n          To use these credentials, unset this environment variable before\n          running your application.\n          '.format(envvar=client.GOOGLE_APPLICATION_CREDENTIALS, override_file=override_file, default_file=config.ADCFilePath()))
        console_io.PromptContinue(message=message, throw_if_unattended=True, cancel_on_no=True)