from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_arguments
def _AddUpgradeSoakingOverrideFlags(self, group: parser_arguments.ArgumentInterceptor):
    """Adds upgrade soaking override flags.

    Args:
      group: The group that should contain the upgrade soaking override flags.
    """
    group = group.add_group(help='        Upgrade soaking override.\n\n        Defines a specific soaking time override for a particular upgrade\n        propagating through the current fleet that supercedes the default\n        soaking duration configured by `--default-upgrade-soaking`.\n\n        To set an upgrade soaking override of 12 hours for the upgrade with\n        name, `k8s_control_plane`, and version, `1.23.1-gke.1000`, run:\n\n          $ {command}               --add-upgrade-soaking-override=12h               --upgrade-selector=name="k8s_control_plane",version="1.23.1-gke.1000"\n        ')
    self._AddAddUpgradeSoakingOverrideFlag(group)
    self._AddUpgradeSelectorFlag(group)