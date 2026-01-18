from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.command_lib.util.resource_map.base import ResourceMapBase
def _register_paths(self):
    self._map_file_path = _RESOURCE_MAP_PATH
    self._schema_file_path = _RESOURCE_SCHEMA_PATH