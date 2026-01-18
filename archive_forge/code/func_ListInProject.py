from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.container.binauthz import apis
from googlecloudsdk.api_lib.container.binauthz import attestors
from googlecloudsdk.api_lib.container.binauthz import containeranalysis
from googlecloudsdk.api_lib.container.binauthz import containeranalysis_apis as ca_apis
from googlecloudsdk.api_lib.container.binauthz import util as binauthz_api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.container.binauthz import flags
from googlecloudsdk.command_lib.container.binauthz import util as binauthz_command_util
from googlecloudsdk.core import resources
def ListInProject(self, args, artifact_digest):
    drydock_client = containeranalysis.Client(ca_apis.GetApiVersion(self.ReleaseTrack()))
    return drydock_client.YieldAttestations(note_ref=None, project_ref=binauthz_api_util.GetProjectRef(), artifact_digest=artifact_digest, page_size=args.page_size, limit=args.limit)