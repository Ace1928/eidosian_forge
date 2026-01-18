from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddInitialRolloutLabelsFlag():
    """Add --initial-rollout-labels flag."""
    help_text = textwrap.dedent('  Labels to apply to the initial rollout when creating the release. Labels take\n  the form of key/value string pairs.\n\n  Examples:\n\n  Add labels:\n\n    $ {command} initial-rollout-labels="commit=abc123,author=foo"\n\n')
    return base.Argument('--initial-rollout-labels', help=help_text, metavar='KEY=VALUE', type=arg_parsers.ArgDict())