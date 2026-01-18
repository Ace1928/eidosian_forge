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
def RegisterInstanceSelectionsPatchEncoders(messages):
    """Registers encoders that will handle null values for Instance Selections in InstanceFlexibilityPolicy message."""

    def _InstanceSelectionsValueEncoder(message):
        """Encoder for Instance Selections map entries.

    It works around issues with proto encoding of InstanceSelections
    with null values by directly encoding a dict of keys with None values into
    json, skipping proto-based encoding.

    Args:
      message: an instance of
        InstanceGroupManagerInstanceFlexibilityPolicy.InstanceSelectionsValue

    Returns:
      JSON string with null value.
    """

        def _GetInstanceSelectionValueOrNone(prop):
            if prop.value is None:
                return None
            return {'rank': prop.value.rank, 'machineTypes': prop.value.machineTypes}
        return json.dumps({property.key: _GetInstanceSelectionValueOrNone(property) for property in message.additionalProperties})

    def _InstanceSelectionsDecoder(data):
        """Decoder for Instance Selections map entries.

    Args:
      data: JSON representation of Instance Selections.

    Returns:
      Instance of
      InstanceGroupManagerInstanceFlexibilityPolicy.InstanceSelectionsValue.
    """
        instance_selections_value = messages.InstanceGroupManagerInstanceFlexibilityPolicy.InstanceSelectionsValue
        py_object = json.loads(data)
        return instance_selections_value(additionalProperties=[instance_selections_value.AdditionalProperty(key=key, value=value) for key, value in py_object.items()])
    encoding.RegisterCustomMessageCodec(encoder=_InstanceSelectionsValueEncoder, decoder=_InstanceSelectionsDecoder)(messages.InstanceGroupManagerInstanceFlexibilityPolicy.InstanceSelectionsValue)