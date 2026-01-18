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
def WriteConfigYaml(collection, output_root, resource_data, release_tracks, enable_overwrites):
    """Writes <comand|spec|test> declarative command files for collection.

  Args:
    collection: Name of collection to generate commands for.
    output_root: Path to the root of the directory. Should just be $PWD when
      executing the `meta generate-config-commands` command.
    resource_data: Resource map data for the given resource.
    release_tracks: Release tracks to generate files for.
    enable_overwrites: True to enable overwriting of existing config export
      files.
  """
    log.status.Print('[{}]:'.format(collection))
    collection_info = resources.REGISTRY.GetCollectionInfo(collection)
    _RenderSurfaceSpecFiles(output_root, resource_data, collection_info, release_tracks, enable_overwrites)
    _RenderCommandGroupInitFile(output_root, resource_data, collection_info, release_tracks, enable_overwrites)
    _RenderCommandFile(output_root, resource_data, collection_info, release_tracks, enable_overwrites)
    _RenderTestFiles(output_root, resource_data, collection_info, enable_overwrites)