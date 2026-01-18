from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import os.path
from googlecloudsdk.core import branding
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import name_parsing
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import files
from mako import runtime
from mako import template
def _MakeResourceArgFlags(collection_info, resource_data):
    """Makes input resource arg flags for config export test file."""
    resource_arg_flags = []
    if getattr(collection_info, 'flat_paths'):
        if '' in getattr(collection_info, 'flat_paths', None):
            components = collection_info.flat_paths[''].split('/')
            resource_arg_flag_names = [component.replace('{', '').replace('Id}', '') for component in components if '{' in component]
            filtered_resource_arg_flag_names = [resource_arg for resource_arg in resource_arg_flag_names if 'project' not in resource_arg]
            formatted_resource_arg_flag_names = []
            for resource_arg in filtered_resource_arg_flag_names[:-1]:
                formatted_name = name_parsing.split_name_on_capitals(name_parsing.singularize(resource_arg), delimiter='-').lower()
                formatted_resource_arg_flag_names.append(formatted_name)
            if 'resource_attribute_renames' in resource_data:
                for original_attr_name, new_attr_name in resource_data.resource_attribute_renames.items():
                    for x in range(len(formatted_resource_arg_flag_names)):
                        if formatted_resource_arg_flag_names[x] == original_attr_name:
                            formatted_resource_arg_flag_names[x] = new_attr_name
            resource_arg_flags = ['--{param}=my-{param}'.format(param=resource_arg) for resource_arg in formatted_resource_arg_flag_names]
    elif getattr(collection_info, 'params', None):
        for param in collection_info.params:
            modified_param_name = param
            if modified_param_name[-2:] == 'Id':
                modified_param_name = modified_param_name[:-2]
            modified_param_name = name_parsing.convert_collection_name_to_delimited(modified_param_name, delimiter='-', make_singular=False)
            if modified_param_name not in (name_parsing.convert_collection_name_to_delimited(collection_info.name, delimiter='-'), 'project', 'name'):
                resource_arg = '--{param}=my-{param}'.format(param=modified_param_name)
                resource_arg_flags.append(resource_arg)
    return ' '.join(resource_arg_flags)