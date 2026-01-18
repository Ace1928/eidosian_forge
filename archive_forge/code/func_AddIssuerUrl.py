from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddIssuerUrl(parser, required=False):
    parser.add_argument('--issuer-url', required=required, help='Issuer url of the cluster to attach.')