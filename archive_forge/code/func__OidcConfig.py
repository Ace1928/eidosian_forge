from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import client
from googlecloudsdk.api_lib.container.gkemulticloud import update_mask
from googlecloudsdk.command_lib.container.attached import flags as attached_flags
from googlecloudsdk.command_lib.container.gkemulticloud import flags
def _OidcConfig(self, args):
    kwargs = {'issuerUrl': attached_flags.GetIssuerUrl(args)}
    oidc = attached_flags.GetOidcJwks(args)
    if oidc:
        kwargs['jwks'] = oidc.encode(encoding='utf-8')
    return self._messages.GoogleCloudGkemulticloudV1AttachedOidcConfig(**kwargs) if any(kwargs.values()) else None