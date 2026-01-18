from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
import six
def AddTemplateParamArgs(parser):
    """Adds --param and --param-from-file flags."""
    parser.add_argument('--param', type=arg_parsers.ArgDict(min_length=1), help='A list of key=value parameters to substitute in the template before the template is submitted to the replica pool. This does not change the actual template file.', metavar='PARAM=VALUE')
    parser.add_argument('--param-from-file', type=arg_parsers.ArgDict(min_length=1), help='A list of files each containing a key=value parameter to substitute in the template before the template is submitted to the replica pool. This does not change the actual template file.', metavar='PARAM=FILE_PATH')