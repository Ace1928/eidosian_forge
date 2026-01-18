from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.compute import constants as compute_constants
from googlecloudsdk.api_lib.container import api_adapter
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.container import constants
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from Google Kubernetes Engine labels that are used for the purpose of tracking
from the node pool, depending on whether locations are being added or removed.
def ValidateCloudRunConfigUpdateArgs(cloud_run_config_args, update_addons_args):
    """Validates flags specifying Cloud Run config for update.

  Args:
    cloud_run_config_args: parsed comandline arguments for --cloud_run_config.
    update_addons_args: parsed comandline arguments for --update-addons.

  Raises:
    InvalidArgumentException: when load-balancer-type is not MTLS_PERMISSIVE nor
    MTLS_STRICT,
    or --update-addons=CloudRun=ENABLED is not specified
  """
    if cloud_run_config_args:
        load_balancer_type = cloud_run_config_args.get('load-balancer-type', '')
        if load_balancer_type not in ['EXTERNAL', 'INTERNAL']:
            raise exceptions.InvalidArgumentException('--kuberun-config', 'load-balancer-type must be one of EXTERNAL or INTERNAL e.g. --kuberun-config load-balancer-type=EXTERNAL')
        if any([update_addons_args.get(v) or False for v in api_adapter.CLOUDRUN_ADDONS]):
            raise exceptions.InvalidArgumentException('--kuberun-config', '--update-addons=KubeRun=ENABLED must be specified when --kuberun-config is given')