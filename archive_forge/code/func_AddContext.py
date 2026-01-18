from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddContext(parser):
    help_txt = 'Context to use in the kubeconfig.'
    parser.add_argument('--context', required=True, help=help_txt)