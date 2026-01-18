from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddInstance(parser, help_text='Secure Source Manager instance used to create the repo'):
    parser.add_argument('--instance', dest='instance', required=True, help=help_text)