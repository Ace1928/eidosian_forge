from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddActivationPolicylag(parser):
    """Adds a --activation-policy flag to the given parser."""
    help_text = "    Activation policy specifies when the instance is activated; it is\n    applicable only when the instance state is 'RUNNABLE'. Valid values:\n\n    ALWAYS: The instance is on, and remains so even in the absence of\n    connection requests.\n\n    NEVER: The instance is off; it is not activated, even if a connection\n    request arrives.\n    "
    choices = ['ALWAYS', 'NEVER']
    parser.add_argument('--activation-policy', help=help_text, choices=choices)