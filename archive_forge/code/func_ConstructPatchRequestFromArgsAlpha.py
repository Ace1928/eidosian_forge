from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.parser_errors import DetailedArgumentError
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import properties
def ConstructPatchRequestFromArgsAlpha(alloydb_messages, instance_ref, args):
    """Constructs the request to update an AlloyDB instance."""
    instance_resource, paths = ConstructInstanceAndUpdatePathsFromArgsAlpha(alloydb_messages, instance_ref, args)
    mask = ','.join(paths) if paths else None
    return alloydb_messages.AlloydbProjectsLocationsClustersInstancesPatchRequest(instance=instance_resource, name=instance_ref.RelativeName(), updateMask=mask)