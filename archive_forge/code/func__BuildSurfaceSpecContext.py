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
def _BuildSurfaceSpecContext(collection_info, release_tracks, resource_data):
    """Makes context dictionary for surface spec rendering."""
    surface_spec_dict = {}
    surface_spec_dict['release_tracks'] = _GetReleaseTracks(release_tracks)
    if 'surface_spec_resource_name' in resource_data:
        surface_spec_dict['surface_spec_resource_arg'] = resource_data.surface_spec_resource_name
    elif 'resource_spec_path' in resource_data:
        surface_spec_dict['surface_spec_resource_arg'] = resource_data.resource_spec_path.split(':')[-1].upper()
    else:
        surface_spec_dict['surface_spec_resource_arg'] = _MakeSurfaceSpecResourceArg(collection_info)
    return surface_spec_dict