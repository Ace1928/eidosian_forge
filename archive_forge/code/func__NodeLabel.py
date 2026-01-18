from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from typing import List
from googlecloudsdk.api_lib.run.integrations import types_utils
from googlecloudsdk.command_lib.run.integrations.typekits import base
from googlecloudsdk.generated_clients.apis.runapps.v1alpha1 import runapps_v1alpha1_messages as runapps
def _NodeLabel(res_id: runapps.ResourceID) -> str:
    type_metadata = types_utils.GetTypeMetadataByResourceType(res_id.type)
    type_name = res_id.type.capitalize()
    if type_metadata and type_metadata.label:
        type_name = type_metadata.label
    return _LABEL_FORMAT.format(type=type_name, name=res_id.name)