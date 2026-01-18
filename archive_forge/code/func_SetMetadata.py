from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.core.resource import resource_property
def SetMetadata(messages, message, resource_type, annotations=None, labels=None):
    """Sets the metadata of a cloud deploy resource message.

  Args:
   messages: module containing the definitions of messages for Cloud Deploy.
   message: message in googlecloudsdk.generated_clients.apis.clouddeploy.
   resource_type: ResourceType enum, the type of the resource to be updated,
     which is defined in the API proto.
   annotations: dict[str,str], a dict of annotation (key,value) pairs that allow
     clients to store small amounts of arbitrary data in cloud deploy resources.
   labels: dict[str,str], a dict of label (key,value) pairs that can be used to
     select cloud deploy resources and to find collections of cloud deploy
     resources that satisfy certain conditions.
  """
    if annotations:
        annotations_value_msg = getattr(messages, resource_type.value).AnnotationsValue
        annotations_value = annotations_value_msg()
        for key, value in annotations.items():
            annotations_value.additionalProperties.append(annotations_value_msg.AdditionalProperty(key=key, value=value))
        message.annotations = annotations_value
    if labels:
        labels_value_msg = getattr(messages, resource_type.value).LabelsValue
        labels_value = labels_value_msg()
        for key, value in labels.items():
            labels_value.additionalProperties.append(labels_value_msg.AdditionalProperty(key=resource_property.ConvertToSnakeCase(key), value=value))
        message.labels = labels_value