from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core import yaml_validator
from googlecloudsdk.core.util import files
import six
def _load_resource_map(self):
    """Loads the ~/resource_map.yaml file into self._resource_map_data."""
    try:
        with files.FileReader(self._map_file_path) as f:
            self._resource_map_data = yaml.load(f)
        if not self._resource_map_data:
            self._resource_map_data = {}
    except files.MissingFileError as err:
        raise ResourceMapInitializationError(err)