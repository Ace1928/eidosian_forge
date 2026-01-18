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
def _BuildYamlMappingTemplateFromMessage(message_cls):
    """Create a stub Apitools To KRM mapping object from a message object."""
    mapping_object = collections.OrderedDict()
    for field in sorted(message_cls.all_fields(), key=lambda x: x.name):
        if field.variant == messages.Variant.MESSAGE:
            fld_map = collections.OrderedDict()
            fld_map['yaml_path'] = _YAML_MAPPING_PLACEHOLDER
            if field.repeated:
                fld_map['repeatable'] = True
            fld_map['submessage_template'] = _BuildYamlMappingTemplateFromMessage(message_cls=field.type)
            mapping_object[field.name] = fld_map
        elif field.repeated:
            fld_map = collections.OrderedDict()
            fld_map['yaml_path'] = _YAML_MAPPING_PLACEHOLDER
            fld_map['repeatable'] = True
            mapping_object[field.name] = fld_map
        else:
            mapping_object[field.name] = _YAML_MAPPING_PLACEHOLDER
    return mapping_object