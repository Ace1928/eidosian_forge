from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.base import ReleaseTrack
from googlecloudsdk.command_lib.assured import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def _AddResourceSettingsFlag(parser, release_track):
    """Adds the resource settings flag to the assured workloads create command.

  Args:
    parser: Parser, Parser used to construct the command flags.
    release_track: ReleaseTrack, Release track of the command being called.

  Returns:
    None.
  """
    if release_track == ReleaseTrack.GA:
        parser.add_argument('--resource-settings', type=arg_parsers.ArgDict(spec={'consumer-project-id': str, 'consumer-project-name': str, 'encryption-keys-project-id': str, 'encryption-keys-project-name': str, 'keyring-id': str}), metavar='KEY=VALUE', help='A comma-separated, key=value map of custom resource settings such as custom project ids, for example: consumer-project-id={CONSUMER_PROJECT_ID} Note: Currently only consumer-project-id, consumer-project-name, encryption-keys-project-id, encryption-keys-project-name and keyring-id are supported. The encryption-keys-project-id, encryption-keys-project-name and keyring-id settings can be specified only if KMS settings are provided')
    else:
        parser.add_argument('--resource-settings', type=arg_parsers.ArgDict(spec={'encryption-keys-project-id': str, 'encryption-keys-project-name': str, 'keyring-id': str}), metavar='KEY=VALUE', help='A comma-separated, key=value map of custom resource settings such as custom project ids, for example: consumer-project-id={CONSUMER_PROJECT_ID} Note: Currently only encryption-keys-project-id, encryption-keys-project-name and keyring-id are supported. The encryption-keys-project-id, encryption-keys-project-name and keyring-id settings can be specified only if KMS settings are provided')