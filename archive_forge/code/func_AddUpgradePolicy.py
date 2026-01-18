from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.container.gkeonprem import flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def AddUpgradePolicy(parser: parser_arguments.ArgumentInterceptor):
    upgrade_policy_group = parser.add_group(help='Upgrade policy for the cluster.')
    upgrade_policy_group.add_argument('--upgrade-policy', type=arg_parsers.ArgDict(spec={'control-plane-only': arg_parsers.ArgBoolean()}), help=textwrap.dedent('      If not specified, control-plane-only is set to False. In the next upgrade operation, all worker node pools will be upgraded together with the control plane.\n\n      Example:\n\n        To upgrade the control plane only and keep worker node pools version unchanged, first specify the policy:\n\n          ```shell\n          $ {command} CLUSTER --upgrade-policy control-plane-only=True\n          ```\n\n        Then to start the upgrade operation using the specified policy, run:\n\n          ```shell\n          $ {parent_command} upgrade CLUSTER --version=VERSION\n          ```\n\n        After upgrading only the cluster control plane, to upgrade an individual node pool, run:\n\n          ```shell\n          $ {grandparent_command} node-pools update NODE_POOL --version=VERSION\n          ```\n\n      Example:\n\n        Alternatively, to upgrade both the control plane and all worker node pools, first specify the policy:\n\n          ```shell\n          $ {command} CLUSTER --upgrade-policy control-plane-only=False\n          ```\n\n        Then to start the upgrade operation using the specified policy, run:\n\n          ```shell\n          $ {parent_command} upgrade CLUSTER --version=VERSION\n          ```\n      '))