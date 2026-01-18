from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddRedirectOptions(parser):
    """Adds redirect action related argument to the argparse."""
    redirect_type = ['google-recaptcha', 'external-302']
    parser.add_argument('--redirect-type', choices=redirect_type, type=lambda x: x.lower(), help="      Type for the redirect action. Default to ``external-302'' if unspecified\n      while --redirect-target is given.\n      ")
    parser.add_argument('--redirect-target', help="      URL target for the redirect action. Must be specified if the redirect\n      type is ``external-302''. Cannot be specified if the redirect type is\n      ``google-recaptcha''.\n      ")