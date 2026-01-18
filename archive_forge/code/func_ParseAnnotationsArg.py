from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
import six
def ParseAnnotationsArg(annotations=None, resource_type=None):
    """Parses and creates the annotations object from the parsed arguments.

  Args:
    annotations: dict, key-value pairs passed in from the --annotations flag.
    resource_type: string, the type of the resource to be created or updated.

  Returns:
    A message object depending on resource_type.

    Service.AnnotationsValue message when resource_type='service' and
    Endpoint.AnnotationsValue message when resource_type='endpoint'.
  """
    if not annotations:
        return None
    msgs = apis.GetMessagesModule(_API_NAME, _VERSION_MAP.get(base.ReleaseTrack.GA))
    additional_properties = []
    if resource_type == 'endpoint':
        annotations_value_msg = msgs.Endpoint.AnnotationsValue
    elif resource_type == 'service':
        annotations_value_msg = msgs.Service.AnnotationsValue
    else:
        return None
    for key, value in six.iteritems(annotations):
        additional_properties.append(annotations_value_msg.AdditionalProperty(key=key, value=value))
    return annotations_value_msg(additionalProperties=additional_properties)