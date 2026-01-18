from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.container.binauthz import arg_parsers
from googlecloudsdk.command_lib.kms import flags as kms_flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs as presentation_specs_lib
def AddDockerCredsArgs(parser):
    """Adds the docker creds args to parser."""
    docker_args_group = parser.add_group(mutex=False, required=False)
    docker_args_group.add_argument('--use-docker-creds', required=False, action='store_true', default=False, help='Whether to use the configuration file where Docker saves authentication credentials when uploading attestations to the registry. If this flag is not passed, or valid credentials are not found, an OAuth2 token for the active gcloud account is used. See https://cloud.google.com/artifact-registry/docs/docker/authentication for more information.')
    docker_args_group.add_argument('--docker-config-dir', required=False, help='Override the directory where the Docker configuration file is searched for. Credentials are pulled from the config.json file under this directory. Defaults to $HOME/.docker.')