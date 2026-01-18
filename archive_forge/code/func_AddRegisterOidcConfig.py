from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddRegisterOidcConfig(parser):
    group = parser.add_mutually_exclusive_group('OIDC config', required=True)
    AddIssuerUrl(group)
    AddHasPrivateIssuer(group)