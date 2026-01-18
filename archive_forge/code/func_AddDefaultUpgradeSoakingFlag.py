from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_arguments
def AddDefaultUpgradeSoakingFlag(self):
    """Adds the --default-upgrade-soaking flag."""
    self.parser.add_argument('--default-upgrade-soaking', type=arg_parsers.Duration(), help='        Configures the default soaking duration for each upgrade propagating\n        through the current fleet to become "COMPLETE". Soaking begins after\n        all clusters in the fleet are on the target version, or after 30 days\n        if all cluster upgrades are not complete. Once an upgrade state becomes\n        "COMPLETE", it will automatically be propagated to the downstream\n        fleet. Max is 30 days.\n\n        To configure Rollout Sequencing for a fleet, this attribute must be\n        set. To do this while specifying a default soaking duration of 7 days,\n        run:\n\n          $ {command} --default-upgrade-soaking=7d\n        ')