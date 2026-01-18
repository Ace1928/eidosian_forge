from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_arguments
def _AddUpgradeSelectorFlag(self, group: parser_arguments.ArgumentInterceptor):
    """Adds the --ugprade-selector flag.

    Args:
      group: The group that should contain the flag.
    """
    group.add_argument('--upgrade-selector', type=UpgradeSelector(), required=True, help='        Name and version of the upgrade to be overridden where version is a\n        full GKE version. Currently, name can be either `k8s_control_plane` or\n        `k8s_node`.\n        ')