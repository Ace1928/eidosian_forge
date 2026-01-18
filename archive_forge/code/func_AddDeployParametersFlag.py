from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddDeployParametersFlag(parser, hidden=False):
    """Add --deploy-parameters flag."""
    help_text = textwrap.dedent('  Deployment parameters to apply to the release. Deployment parameters take the form of key/value string pairs.\n\n  Examples:\n\n  Add deployment parameters:\n\n    $ {command} --deploy-parameters="key1=value1,key2=value2"\n\n')
    parser.add_argument('--deploy-parameters', metavar='KEY=VALUE', type=arg_parsers.ArgDict(), hidden=hidden, help=help_text)