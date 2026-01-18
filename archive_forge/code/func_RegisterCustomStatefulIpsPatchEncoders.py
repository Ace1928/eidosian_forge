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
def RegisterCustomStatefulIpsPatchEncoders(client):
    """Registers decoders and encoders that will handle null values for Internal and External IPs maps in StatefulPolicy message."""
    auto_delete_map = {'NEVER': client.messages.StatefulPolicyPreservedStateNetworkIp(autoDelete=client.messages.StatefulPolicyPreservedStateNetworkIp.AutoDeleteValueValuesEnum.NEVER), 'ON_PERMANENT_INSTANCE_DELETION': client.messages.StatefulPolicyPreservedStateNetworkIp(autoDelete=client.messages.StatefulPolicyPreservedStateNetworkIp.AutoDeleteValueValuesEnum.ON_PERMANENT_INSTANCE_DELETION)}

    def _StatefulIpsValueEncoder(message):
        """Encoder for Stateful Ips map entries.

    It works around issues with proto encoding of StatefulPolicyPreservedState
    with null values by directly encoding a dict of keys with None values into
    json, skipping proto-based encoding.

    Args:
      message: an instance of StatefulPolicyPreservedState.InternalIPsValue or
        StatefulPolicyPreservedState.ExternalIPsValue

    Returns:
      JSON string with null value.
    """
        return json.dumps({property.key: _GetAutodeleteOrNone(property) for property in message.additionalProperties})

    def _GetAutodeleteOrNone(autodelete):
        if autodelete.value is None:
            return None
        return {'autoDelete': autodelete.value.autoDelete.name}

    def _InternalStatefulIpsDecoder(data):
        """Decoder for Stateful Ips map entries.

    Args:
      data: JSON representation of Internal Stateful Ips.

    Returns:
      Instance of StatefulPolicyPreservedState.InternalIPsValue.
    """
        ips_value = client.messages.StatefulPolicyPreservedState.InternalIPsValue
        return _CommonStatefulIpsDecoder(data, ips_value)

    def _ExternalStatefulIpsDecoder(data):
        """Decoder for Stateful Ips map entries.

    Args:
      data: JSON representation of External Stateful Ips.

    Returns:
      Instance of
      StatefulPolicyPreservedState.ExternalIPsValue.AdditionalProperty.
    """
        ips_value = client.messages.StatefulPolicyPreservedState.ExternalIPsValue
        return _CommonStatefulIpsDecoder(data, ips_value)

    def _CommonStatefulIpsDecoder(data, ips_value):
        py_object = json.loads(data)
        return ips_value(additionalProperties=[ips_value.AdditionalProperty(key=key, value=auto_delete_map[value['autoDelete']]) for key, value in py_object.items()])
    encoding.RegisterCustomMessageCodec(encoder=_StatefulIpsValueEncoder, decoder=_InternalStatefulIpsDecoder)(client.messages.StatefulPolicyPreservedState.InternalIPsValue)
    encoding.RegisterCustomMessageCodec(encoder=_StatefulIpsValueEncoder, decoder=_ExternalStatefulIpsDecoder)(client.messages.StatefulPolicyPreservedState.ExternalIPsValue)