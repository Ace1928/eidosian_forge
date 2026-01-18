from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.secrets import api as secrets_api
from googlecloudsdk.api_lib.util import exceptions
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.secrets import args as secrets_args
from googlecloudsdk.command_lib.secrets import fmt as secrets_fmt
from googlecloudsdk.command_lib.secrets import util as secrets_util
from googlecloudsdk.command_lib.util import crc32c
@base.ReleaseTracks(base.ReleaseTrack.GA)
class Access(base.DescribeCommand):
    """Access a secret version's data.

  Access the data for the specified secret version.

  ## EXAMPLES

  Access the data for version 123 of the secret 'my-secret':

    $ {command} 123 --secret=my-secret

  Note: The output will be formatted as UTF-8 which can corrupt binary secrets.

  To write raw bytes to a file use --out-file flag:

    $ {command} 123 --secret=my-secret --out-file=/tmp/secret

  To get the raw bytes, have Google Cloud CLI print the response as
  base64-encoded and decode:

    $ {command} 123 --secret=my-secret --format='get(payload.data)' | tr '_-' '/+' | base64 -d
  """
    EMPTY_OUT_FILE_MESSAGE = 'The value provided for --out-file is the empty string. This can happen if you pass or pipe a variable that is undefined. Please verify that the --out-file flag is not the empty string.'

    @staticmethod
    def Args(parser):
        secrets_args.AddVersionOrAlias(parser, purpose='to access', positional=True, required=True)
        secrets_args.AddOutFile(parser)
        secrets_fmt.UseSecretData(parser)

    def Run(self, args):
        version_ref = args.CONCEPTS.version.Parse()
        version = secrets_api.Versions().Access(version_ref)
        if version.payload.dataCrc32c is None or crc32c.does_data_match_checksum(version.payload.data, version.payload.dataCrc32c):
            if args.IsSpecified('out_file'):
                if not args.out_file:
                    raise calliope_exceptions.BadFileException(self.EMPTY_OUT_FILE_MESSAGE)
                args.format = 'disable'
                secrets_util.WriteBinaryFile(args.out_file, version.payload.data)
            return version
        raise exceptions.HttpException(CHECKSUM_VERIFICATION_FAILURE_MESSAGE)