from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from typing import List
from googlecloudsdk.api_lib.run.integrations import types_utils
from googlecloudsdk.command_lib.run.integrations.typekits import base
from googlecloudsdk.generated_clients.apis.runapps.v1alpha1 import runapps_v1alpha1_messages as runapps
def _GraphvizEdge(from_id, to_id):
    return _EDGE_FORMAT.format(edge_source=_NodeName(from_id), edge_dest=_NodeName(to_id))