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
def _export_resource_map(self, file_path=None, prune=False, validate=True):
    """Prunes and exports self._resource_map_data to ~/resource_map.yaml."""
    try:
        if prune:
            self.prune()
        if validate:
            self._validate_resource_map()
        with files.FileWriter(file_path or self._map_file_path) as f:
            yaml.dump(self._resource_map_data, stream=f)
    except files.MissingFileError as err:
        raise ResourceMapInitializationError(err)