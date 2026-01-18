from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddEnableInitialRolloutFlag():
    """Adds --enable-initial-rollout flag."""
    return base.Argument('--enable-initial-rollout', action='store_const', help='Creates a rollout in the first target defined in the delivery pipeline. This is the default behavior.', const=True)