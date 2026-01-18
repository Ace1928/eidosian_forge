from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import json
from googlecloudsdk.api_lib.artifacts import exceptions as ar_exceptions
from googlecloudsdk.api_lib.auth import service_account
from googlecloudsdk.command_lib.artifacts import requests as ar_requests
from googlecloudsdk.command_lib.artifacts import util as ar_util
from googlecloudsdk.command_lib.artifacts.print_settings import apt
from googlecloudsdk.command_lib.artifacts.print_settings import gradle
from googlecloudsdk.command_lib.artifacts.print_settings import mvn
from googlecloudsdk.command_lib.artifacts.print_settings import npm
from googlecloudsdk.command_lib.artifacts.print_settings import python
from googlecloudsdk.command_lib.artifacts.print_settings import yum
from googlecloudsdk.core import config
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.credentials import creds
from googlecloudsdk.core.credentials import exceptions as creds_exceptions
from googlecloudsdk.core.credentials import store
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
def GetMavenTemplate(messages, maven_cfg, sa_creds):
    """Forms a maven snippet to add to the pom.xml file.

  Args:
    messages: Module, the messages module for the API.
    maven_cfg: MavenRepositoryConfig, the maven configuration proto that
      contains the version policy.
    sa_creds: str, service account credentials.

  Returns:
    str, a maven template to add to pom.xml.
  """
    mvn_template = mvn.NO_SERVICE_ACCOUNT_TEMPLATE
    if maven_cfg and maven_cfg.versionPolicy == messages.MavenRepositoryConfig.VersionPolicyValueValuesEnum.SNAPSHOT:
        mvn_template = mvn.NO_SERVICE_ACCOUNT_SNAPSHOT_TEMPLATE
        if sa_creds:
            mvn_template = mvn.SERVICE_ACCOUNT_SNAPSHOT_TEMPLATE
    elif maven_cfg and maven_cfg.versionPolicy == messages.MavenRepositoryConfig.VersionPolicyValueValuesEnum.RELEASE:
        mvn_template = mvn.NO_SERVICE_ACCOUNT_RELEASE_TEMPLATE
        if sa_creds:
            mvn_template = mvn.SERVICE_ACCOUNT_RELEASE_TEMPLATE
    elif sa_creds:
        mvn_template = mvn.SERVICE_ACCOUNT_TEMPLATE
    return mvn_template