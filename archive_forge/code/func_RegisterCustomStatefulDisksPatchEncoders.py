from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import random
import re
import string
import sys
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.api_lib.compute import lister
from googlecloudsdk.api_lib.compute import path_simplifier
from googlecloudsdk.api_lib.compute import request_helper
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.managed_instance_groups import auto_healing_utils
from googlecloudsdk.command_lib.compute.managed_instance_groups import update_instances_utils
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
import six
from six.moves import range  # pylint: disable=redefined-builtin
def RegisterCustomStatefulDisksPatchEncoders(client):
    """Registers decoders and encoders that will handle null values for Stateful Disks map in StatefulPolicy message."""
    auto_delete_map = {'NEVER': client.messages.StatefulPolicyPreservedStateDiskDevice(autoDelete=client.messages.StatefulPolicyPreservedStateDiskDevice.AutoDeleteValueValuesEnum.NEVER), 'ON_PERMANENT_INSTANCE_DELETION': client.messages.StatefulPolicyPreservedStateDiskDevice(autoDelete=client.messages.StatefulPolicyPreservedStateDiskDevice.AutoDeleteValueValuesEnum.ON_PERMANENT_INSTANCE_DELETION)}

    def _StatefulDisksValueEncoder(message):
        """Encoder for Stateful Disks map entries.

    It works around issues with proto encoding of StatefulPolicyPreservedState
    with null values by directly encoding a dict of keys with None values into
    json, skipping proto-based encoding.

    Args:
      message: an instance of StatefulPolicyPreservedState.DisksValue

    Returns:
      JSON string with null value.
    """
        return json.dumps({property.key: _GetAutodeleteOrNone(property) for property in message.additionalProperties})

    def _GetAutodeleteOrNone(autodelete):
        if autodelete.value is None:
            return None
        return {'autoDelete': autodelete.value.autoDelete.name}

    def _StatefulDisksDecoder(data):
        """Decoder for Stateful Disks map entries.

    Args:
      data: JSON representation of Stateful Disks.

    Returns:
      Instance of StatefulPolicyPreservedState.DisksValue.
    """
        disk_device_value = client.messages.StatefulPolicyPreservedState.DisksValue
        py_object = json.loads(data)
        return disk_device_value(additionalProperties=[disk_device_value.AdditionalProperty(key=key, value=auto_delete_map[value['autoDelete']]) for key, value in py_object.items()])
    encoding.RegisterCustomMessageCodec(encoder=_StatefulDisksValueEncoder, decoder=_StatefulDisksDecoder)(client.messages.StatefulPolicyPreservedState.DisksValue)