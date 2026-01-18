from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.core.util import files
def GetClientSecretFlag():
    """Anthos auth token client-secret flag, specifies the Client Secret is the shared secret between OIDC client application and OIDC provider."""
    return base.Argument('--client-secret', required=False, help='Client Secret is the shared secret between OIDC client application and OIDC provider.')