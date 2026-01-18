from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddRollbackOfRollout(parser):
    """Add --rollback-of-rollout flag."""
    help_text = textwrap.dedent('  If set, this validates whether the rollout name specified by the flag matches\n  the rollout on the target.\n\n  Examples:\n\n  Validate that `test-rollout` is the rollout to rollback on the target.\n\n    $ {command} --rollback-of-rollout=projects/test-project/locations/us-central1/deliveryPipelines/test-pipeline/releases/test-release/rollouts/test-rollout\n\n  ')
    parser.add_argument('--rollback-of-rollout', help=help_text, hidden=False, default=None, required=False)