from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.secrets import api as secrets_api
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.secrets import args as secrets_args
from googlecloudsdk.command_lib.secrets import log as secrets_log
@base.ReleaseTracks(base.ReleaseTrack.BETA)
class DisableBeta(Disable):
    """Disable the version of the provided secret.

  Disable the version of the provided secret. It can be re-enabled with
  `{parent_command} enable`.

  ## EXAMPLES

  Disable version '123' of the secret named 'my-secret':

    $ {command} 123 --secret=my-secret

  Disable version '123' of the secret named 'my-secret' using an etag:

    $ {command} 123 --secret=my-secret --etag=\\"123\\"
  """

    @staticmethod
    def Args(parser):
        secrets_args.AddVersion(parser, purpose='to disable', positional=True, required=True)
        secrets_args.AddLocation(parser, purpose='to disable', hidden=True)
        secrets_args.AddVersionEtag(parser)

    def Run(self, args):
        api_version = secrets_api.GetApiFromTrack(self.ReleaseTrack())
        version_ref = args.CONCEPTS.version.Parse()
        result = secrets_api.Versions(api_version=api_version).Disable(version_ref, etag=args.etag, secret_location=args.location)
        secrets_log.Versions().Disabled(version_ref)
        return result