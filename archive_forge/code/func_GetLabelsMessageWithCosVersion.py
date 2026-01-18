from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import itertools
import re
import enum
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
def GetLabelsMessageWithCosVersion(labels, image_uri, resources, resource_class):
    """Returns message with labels for instance / instance template.

  Args:
    labels: dict, labels to assign to the resource.
    image_uri: URI of image used as a base for the resource. The function
               extracts COS version from the URI and uses it as a value of
               `container-vm` label.
    resources: object that can parse image_uri.
    resource_class: class of the resource to which labels will be assigned.
                    Must contain LabelsValue class and
                    resource_class.LabelsValue must contain AdditionalProperty
                    class.
  """
    cos_version = resources.Parse(image_uri, collection='compute.images').Name().replace('/', '-')
    if labels is None:
        labels = {}
    labels['container-vm'] = cos_version
    additional_properties = [resource_class.LabelsValue.AdditionalProperty(key=k, value=v) for k, v in sorted(six.iteritems(labels))]
    return resource_class.LabelsValue(additionalProperties=additional_properties)