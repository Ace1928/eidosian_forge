from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
import six
def AddEntityTypeKind(unused_instance_ref, unused_args, request):
    entities = request.googleCloudDialogflowV2EntityType.entities
    enum = request.googleCloudDialogflowV2EntityType.KindValueValuesEnum
    kind = enum.KIND_LIST
    for entity in entities:
        if entity.synonyms != [entity.value]:
            kind = enum.KIND_MAP
    request.googleCloudDialogflowV2EntityType.kind = kind
    return request