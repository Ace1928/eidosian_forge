from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import collections
import os
from apitools.base.protorpclite import messages
from googlecloudsdk.command_lib.anthos.common import file_parsers
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.apis import registry
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core import yaml_validator
from googlecloudsdk.core.util import files
import six
def _MapDictToApiToolsMessage(data, mapping, message):
    """Helper function to do actual KRM to Apitools Mapping."""
    actual_fields = set()
    for field, descriptor in six.iteritems(mapping):
        if file_parsers.FindOrSetItemInDict(data, descriptor.yaml_path):
            actual_fields.add(field)
    if not actual_fields:
        raise InvalidDataError('Input YAML contains no message data')
    output_data = {}
    for field in sorted(message.all_fields(), key=lambda x: x.name):
        if field.name not in actual_fields:
            continue
        mapping_descriptor = mapping[field.name]
        value = file_parsers.FindOrSetItemInDict(data, mapping_descriptor.yaml_path)
        if field.variant == messages.Variant.MESSAGE:
            if field.repeated:
                value = value if yaml.list_like(value) else [value]
                sub_message = []
                for item in value:
                    sub_message.append(ParseMessageFromDict(item, mapping_descriptor.submessage_template, field.type))
            else:
                sub_message = ParseMessageFromDict(value, mapping_descriptor.submessage_template, field.type)
            if sub_message:
                output_data[field.name] = sub_message
        elif field.repeated:
            if yaml.list_like(value):
                output_data[field.name] = [_ParseFieldValue(field, x) for x in value]
            else:
                output_data[field.name] = [_ParseFieldValue(field, value)]
        else:
            output_data[field.name] = _ParseFieldValue(field, value)
    return message(**output_data)