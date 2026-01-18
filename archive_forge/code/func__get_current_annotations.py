from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from typing import Optional
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.container.gkeonprem import client
from googlecloudsdk.api_lib.container.gkeonprem import update_mask
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.core import properties
from googlecloudsdk.generated_clients.apis.gkeonprem.v1 import gkeonprem_v1_messages as messages
def _get_current_annotations(self, args: parser_extensions.Namespace):
    """Fetches the standalone cluster annotations."""
    cluster_ref = args.CONCEPTS.cluster.Parse()
    cluster_response = self.Describe(cluster_ref)
    curr_annotations = {}
    if cluster_response.annotations:
        for annotation in cluster_response.annotations.additionalProperties:
            curr_annotations[annotation.key] = annotation.value
    return curr_annotations