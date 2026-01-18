from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_arguments
def _AddAddUpgradeSoakingOverrideFlag(self, group: parser_arguments.ArgumentInterceptor):
    """Adds the --add-upgrade-soaking-override flag.

    Args:
      group: The group that should contain the flag.
    """
    group.add_argument('--add-upgrade-soaking-override', type=arg_parsers.Duration(), required=True, help='        Overrides the soaking time for a particular upgrade name and version\n        propagating through the current fleet. Set soaking to 0 days to bypass\n        soaking and fast-forward the upgrade to the downstream fleet.\n\n        See `$ gcloud topic datetimes` for information on duration formats.\n        ')