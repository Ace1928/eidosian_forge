from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddOidcJwks(parser):
    parser.add_argument('--oidc-jwks', help='OIDC JWKS of the cluster to attach.')