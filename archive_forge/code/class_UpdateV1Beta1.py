from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.ai import operations
from googlecloudsdk.api_lib.ai.indexes import client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import endpoint_util
from googlecloudsdk.command_lib.ai import flags
from googlecloudsdk.command_lib.ai import indexes_util
from googlecloudsdk.command_lib.ai import operations_util
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
@base.ReleaseTracks(base.ReleaseTrack.BETA, base.ReleaseTrack.ALPHA)
class UpdateV1Beta1(UpdateV1):
    """Update an Vertex AI index.

  ## EXAMPLES

  To update index `123` under project `example` in region `us-central1`, run:

    $ {command} --display-name=new-name
    --metadata-file=path/to/your/metadata.json --project=example
    --region=us-central1
  """

    def Run(self, args):
        return self._Run(args, constants.BETA_VERSION)