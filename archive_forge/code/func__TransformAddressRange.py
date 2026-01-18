from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import lister
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.addresses import flags
import six
def _TransformAddressRange(resource):
    prefix_length = resource.get('prefixLength')
    address = resource.get('address')
    if prefix_length:
        return address + '/' + six.text_type(prefix_length)
    return address