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
def JarMatcher(jar_path, stager, appyaml):
    """Generate a Service from a Java fatjar path.

  This function is a path matcher that returns if and only if:
  - `jar_path` points to  a jar file .

  The service will be staged according to the stager as a jar runtime,
  which is defined in staging.py.

  Args:
    jar_path: str, Unsanitized absolute path pointing to a file of jar type.
    stager: staging.Stager, stager that will be invoked if there is a runtime
      and environment match.
    appyaml: str or None, the app.yaml location to used for deployment.

  Raises:
    staging.StagingCommandFailedError, staging command failed.

  Returns:
    Service, fully populated with entries that respect a staged deployable
        service, or None if the path does not match the pattern described.
  """
    _, ext = os.path.splitext(jar_path)
    if os.path.exists(jar_path) and ext in ['.jar']:
        app_dir = os.path.abspath(os.path.join(jar_path, os.pardir))
        descriptor = jar_path
        staging_dir = stager.Stage(descriptor, app_dir, 'java-jar', env.STANDARD, appyaml)
        yaml_path = os.path.join(staging_dir, 'app.yaml')
        service_info = yaml_parsing.ServiceYamlInfo.FromFile(yaml_path)
        return Service(descriptor, app_dir, service_info, staging_dir)
    return None