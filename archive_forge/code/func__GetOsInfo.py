from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute.os_config import utils as osconfig_api_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instances import flags
from googlecloudsdk.core import log
from googlecloudsdk.core.resource import resource_projector
import six
def _GetOsInfo(self, guest_attributes):
    guest_attributes_json = resource_projector.MakeSerializable(guest_attributes)
    os_info = {}
    for guest_attribute in guest_attributes_json:
        guest_attribute_key = guest_attribute['key']
        if guest_attribute_key in self._OS_INFO_FIELD_KEYS:
            os_info[guest_attribute_key] = guest_attribute['value']
    return os_info