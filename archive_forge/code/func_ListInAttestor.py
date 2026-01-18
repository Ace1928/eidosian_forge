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
def ListInAttestor(self, args, artifact_digest):
    attestors_client = attestors.Client(apis.GetApiVersion(self.ReleaseTrack()))
    drydock_client = containeranalysis.Client(ca_apis.GetApiVersion(self.ReleaseTrack()))
    attestor_ref = args.CONCEPTS.attestor.Parse()
    attestor = attestors_client.Get(attestor_ref)
    note_ref = resources.REGISTRY.ParseResourceId('containeranalysis.projects.notes', attestors_client.GetNoteAttr(attestor).noteReference, {})
    return drydock_client.YieldAttestations(note_ref=note_ref, artifact_digest=artifact_digest, page_size=args.page_size, limit=args.limit)