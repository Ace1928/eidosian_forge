from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.endpoints import config_reporter
from googlecloudsdk.api_lib.endpoints import exceptions
from googlecloudsdk.api_lib.endpoints import services_util
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.api_lib.services import exceptions as services_exceptions
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import http_encoding
import six.moves.urllib.parse
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class DeployBetaAlpha(_BaseDeploy, base.Command):
    """Deploys a service configuration for the given service name.

     This command is used to deploy a service configuration for a service
     to Google Service Management. As input, it takes one or more paths
     to service configurations that should be uploaded. These configuration
     files can be Proto Descriptors, Open API (Swagger) specifications,
     or Google Service Configuration files in JSON or YAML formats.

     If a service name is present in multiple configuration files (given
     in the `host` field in OpenAPI specifications or the `name` field in
     Google Service Configuration files), the first one will take precedence.

     When deploying a new service configuration to an already-existing
     service, some safety checks will be made comparing the new configuration
     to the active configuration. If any actionable advice is provided, it
     will be printed out to the log, and the deployment will be halted. It is
     recommended that these warnings be addressed before proceeding, but they
     can be overridden with the --force flag.

     This command will block until deployment is complete unless the
     `--async` flag is passed.

     ## EXAMPLES
     To deploy a single Open API service configuration, run:

       $ {command} ~/my_app/openapi.json

     To run the deployment asynchronously (non-blocking), run:

       $ {command} ~/my_app/openapi.json --async

     To deploy a service config with a Proto, run:

       $ {command} ~/my_app/service-config.yaml ~/my_app/service-protos.pb
  """

    def CheckPushAdvisor(self, force=False):
        """Run the Push Advisor and return whether the command should abort.

    Args:
      force: bool, if True, this method will return False even if warnings are
        generated.

    Returns:
      True if the deployment should be aborted due to warnings, otherwise
      False if it's safe to continue.
    """
        log_func = log.warning if force else log.error
        num_advices = self.ShowConfigReport(self.service_name, self.service_config_id, log_func)
        if num_advices > 0:
            if force:
                log_func('{0}\n'.format(FORCE_ADVICE_STRING))
            else:
                log_func('{0}\n'.format(ADVICE_STRING))
                return True
        return False