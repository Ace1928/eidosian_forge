from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import os
from googlecloudsdk.api_lib.app import env
from googlecloudsdk.api_lib.app import yaml_parsing
from googlecloudsdk.command_lib.app import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
def ExplicitAppYamlMatcher(path, stager, appyaml):
    """Use optional app.yaml with a directory or a file the user wants to deploy.

  Args:
    path: str, Unsanitized absolute path, may point to a directory or a file of
      any type. There is no guarantee that it exists.
    stager: staging.Stager, stager that will not be invoked.
    appyaml: str or None, the app.yaml location to used for deployment.

  Returns:
    Service, fully populated with entries that respect a staged deployable
        service, or None if there is no optional --appyaml flag usage.
  """
    if appyaml:
        service_info = yaml_parsing.ServiceYamlInfo.FromFile(appyaml)
        staging_dir = stager.Stage(appyaml, path, 'generic-copy', service_info.env, appyaml)
        return Service(appyaml, path, service_info, staging_dir)
    return None