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
def _BuildCommandContext(collection_info, release_tracks, resource_data):
    """Makes context dictionary for config export command template rendering."""
    command_dict = {}
    command_dict['collection_name'] = collection_info.name
    command_dict['branded_api_name'] = branding.Branding().get(collection_info.api_name, collection_info.api_name.capitalize())
    command_dict['plural_resource_name_with_spaces'] = name_parsing.convert_collection_name_to_delimited(collection_info.name, make_singular=False)
    command_dict['singular_name_with_spaces'] = name_parsing.convert_collection_name_to_delimited(collection_info.name)
    command_dict['singular_capitalized_name'] = command_dict['singular_name_with_spaces'].capitalize()
    if 'resource_spec_path' in resource_data:
        command_dict['resource_spec_path'] = resource_data.resource_spec_path
    else:
        resource_spec_name = command_dict['singular_name_with_spaces'].replace(' ', '_')
        resource_spec_dir = resource_data.home_directory.split('.')[0]
        command_dict['resource_spec_path'] = '{}.resources:{}'.format(resource_spec_dir, resource_spec_name)
    command_dict['resource_argument_name'] = _MakeResourceArgName(collection_info.name)
    command_dict['release_tracks'] = _GetReleaseTracks(release_tracks)
    api_a_or_an = 'a'
    if command_dict['branded_api_name'][0] in 'aeiou':
        api_a_or_an = 'an'
    command_dict['api_a_or_an'] = api_a_or_an
    resource_a_or_an = 'a'
    if command_dict['singular_name_with_spaces'][0] in 'aeiou':
        resource_a_or_an = 'an'
    command_dict['resource_a_or_an'] = resource_a_or_an
    return command_dict