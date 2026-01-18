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
def GetDeployables(args, stager, path_matchers, appyaml=None):
    """Given a list of args, infer the deployable services and configs.

  Given a deploy command, e.g. `gcloud app deploy ./dir other/service.yaml
  cron.yaml WEB-INF/appengine-web.xml`, the deployables can be on multiple
  forms. This method pre-processes and infers yaml descriptors from the
  various formats accepted. The rules are as following:

  This function is a context manager, and should be used in conjunction with
  the `with` keyword.

  1. If `args` is an empty list, add the current directory to it.
  2. For each arg:
    - If arg refers to a config file, add it to the configs set.
    - Else match the arg against the path matchers. The first match will win.
      The match will be added to the services set. Matchers may run staging.

  Args:
    args: List[str], positional args as given on the command-line.
    stager: staging.Stager, stager that will be invoked on sources that have
        entries in the stager's registry.
    path_matchers: List[Function], list of functions on the form
        fn(path, stager) ordered by descending precedence, where fn returns
        a Service or None if no match.
    appyaml: str or None, the app.yaml location to used for deployment.

  Raises:
    FileNotFoundError: One or more argument does not point to an existing file
        or directory.
    UnknownSourceError: Could not infer a config or service from an arg.
    DuplicateConfigError: Two or more config files have the same type.
    DuplicateServiceError: Two or more services have the same service id.

  Returns:
    Tuple[List[Service], List[ConfigYamlInfo]], lists of deployable services
    and configs.
  """
    if not args:
        args = ['.']
    paths = [os.path.abspath(arg) for arg in args]
    configs = Configs()
    services = Services()
    if appyaml:
        if len(paths) > 1:
            raise exceptions.MultiDeployError()
        if not os.path.exists(os.path.abspath(appyaml)):
            raise exceptions.FileNotFoundError('File {0} referenced by --appyaml does not exist.'.format(appyaml))
        if not os.path.exists(paths[0]):
            raise exceptions.FileNotFoundError(paths[0])
    for path in paths:
        if not os.path.exists(path):
            raise exceptions.FileNotFoundError(path)
        config = yaml_parsing.ConfigYamlInfo.FromFile(path)
        if config:
            configs.Add(config)
            continue
        service = Service.FromPath(path, stager, path_matchers, appyaml)
        if service:
            services.Add(service)
            continue
        raise exceptions.UnknownSourceError(path)
    return (services.GetAll(), configs.GetAll())