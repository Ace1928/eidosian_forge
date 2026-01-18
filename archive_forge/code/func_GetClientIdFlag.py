from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.core.util import files
def GetClientIdFlag():
    """Anthos auth token client-id flag, specifies the ClientID is the id for OIDC client application."""
    return base.Argument('--client-id', required=False, help='ClientID is the id for OIDC client application.')