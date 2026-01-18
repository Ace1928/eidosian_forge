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
def _RenderSurfaceSpecFiles(output_root, resource_data, collection_info, release_tracks, enable_overwrites):
    """Render surface spec files (both GROUP.yaml and command spec file.)"""
    context = _BuildSurfaceSpecContext(collection_info, release_tracks, resource_data)
    group_template = _BuildTemplate('surface_spec_group_template.tpl')
    group_file_path = _BuildFilePath(output_root, _SPEC_PATH_COMPONENTS, resource_data.home_directory, 'config', 'GROUP.yaml')
    _RenderFile(group_file_path, group_template, context, enable_overwrites)
    spec_path = _BuildFilePath(output_root, _SPEC_PATH_COMPONENTS, resource_data.home_directory, 'config', 'export.yaml')
    spec_template = _BuildTemplate('surface_spec_template.tpl')
    _RenderFile(spec_path, spec_template, context, enable_overwrites)