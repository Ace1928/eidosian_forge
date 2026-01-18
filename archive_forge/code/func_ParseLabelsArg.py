from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
import six
def ParseLabelsArg(labels=None, release_track=base.ReleaseTrack.GA):
    """Parses and creates the labels object from the parsed arguments.

  Args:
    labels: dict, key-value pairs passed in from the --labels flag.
    release_track: base.ReleaseTrack value

  Returns:
    A message object.
  """
    if not labels:
        return None
    msgs = apis.GetMessagesModule(_API_NAME, _VERSION_MAP.get(release_track))
    additional_properties = []
    labels_value_msg = msgs.Namespace.LabelsValue
    for key, value in six.iteritems(labels):
        additional_properties.append(labels_value_msg.AdditionalProperty(key=key, value=value))
    return labels_value_msg(additionalProperties=additional_properties)