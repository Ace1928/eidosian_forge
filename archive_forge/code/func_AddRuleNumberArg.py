from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddRuleNumberArg(parser, operation_type='operate on', plural=False):
    """Adds a positional argument for the Rule number."""
    help_text = 'Number that uniquely identifies the Rule{} to {}'.format('s' if plural else '', operation_type)
    params = {'help': help_text}
    if plural:
        params['nargs'] = '+'
    parser.add_argument('rule_number', type=int, **params)