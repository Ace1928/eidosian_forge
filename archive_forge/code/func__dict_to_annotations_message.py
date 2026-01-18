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
def _dict_to_annotations_message(self, annotations):
    """Converts key-val pairs to proto message AnnotationsValue."""
    additional_property_messages = []
    if not annotations:
        return None
    for key, value in annotations.items():
        additional_property_messages.append(messages.BareMetalCluster.AnnotationsValue.AdditionalProperty(key=key, value=value))
    annotation_value_message = messages.BareMetalCluster.AnnotationsValue(additionalProperties=additional_property_messages)
    return annotation_value_message