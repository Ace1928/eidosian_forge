from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_arguments
def _AddRemoveUpgradeSoakingOverridesFlag(self, group: parser_arguments.ArgumentInterceptor):
    """Adds the --remove-upgrade-soaking-overrides flag.

    Args:
      group: The group that should contain the flag.
    """
    group.add_argument('--remove-upgrade-soaking-overrides', action='store_true', default=None, help='        Removes soaking time overrides for all upgrades propagating through the\n        current fleet. Consequently, all upgrades will follow the soak time\n        configured by `--default-upgrade-soaking` until new overrides are\n        configured with `--add_upgrade_soaking_override` and\n        `--upgrade_selector`.\n\n        To remove all configured soaking time overrides, run:\n\n          $ {command} --remove-upgrade-soaking-overrides\n        ')