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
class ResourceMapBase(six.with_metaclass(abc.ABCMeta)):
    """Base data wrapper class for Resource Map metadata yaml files.

  This object loads the relevant resource map file upon instantiation and sets
  the parsed dictionary as the internal attribute _resource_map_data. Underlying
  dictionary data is never interacted with directly, and is instead is
  set/retrieved/interacted with via an ApiData wrapper object.

  Attributes:
    _resource_map_data: Dict containing metadata for each resource in each api.
  """

    def __init__(self):
        self._map_file_path = None
        self._schema_file_path = None
        self._register_paths()
        self._resource_map_data = {}
        self._load_resource_map()

    def __getattr__(self, api_name):
        """Returns underlying API data when accessing attribute."""
        if api_name.startswith('_'):
            raise PrivateAttributeNotFoundError('ResourceMap', api_name)
        return self.get_api(api_name)

    def __contains__(self, api_name):
        """Returns True if api_name exists in self._resource_map_data."""
        return api_name in self._resource_map_data

    def __iter__(self):
        """Yields ApiData wrapper objects for each API in _resource_map_data."""
        for api_name, api_data in six.iteritems(self._resource_map_data):
            yield ApiData(api_name, api_data)

    def __eq__(self, other):
        return self.to_dict() == other.to_dict()

    @abc.abstractmethod
    def _register_paths(self):
        """Must be overridden by child classes to register map and schema paths.

    Must explicitly set self._map_file_path and self._schema_file_path to
    appropriate filepaths in the overridden method of the child class.
    """
        pass

    def _load_resource_map(self):
        """Loads the ~/resource_map.yaml file into self._resource_map_data."""
        try:
            with files.FileReader(self._map_file_path) as f:
                self._resource_map_data = yaml.load(f)
            if not self._resource_map_data:
                self._resource_map_data = {}
        except files.MissingFileError as err:
            raise ResourceMapInitializationError(err)

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

    def _validate_resource_map(self):
        """Validates resource map against ~/resource_map_schema.yaml."""
        yaml_validator.Validator(self._schema_file_path).Validate(self._resource_map_data)

    def to_dict(self):
        return self._resource_map_data

    def prune(self):
        """Prunes the resource map, removing redundant metadata values in the map.

    Calls prune() on each ApiData wrapper object, which in turn calls prune()
    on each underlying resource. Pruning each resource will remove any instances
    of a track-specific metadata field being set to the same value as the parent
    resource metadata field, eliminating any redundancies and keeping the map
    as clean as possible.
    """
        for api_data in iter(self):
            api_data.prune()

    def get_api(self, api_name):
        """Returns the api data wrapped in an ApiData object."""
        if api_name not in self._resource_map_data:
            raise ApiNotFoundError(api_name)
        return ApiData(api_name, self._resource_map_data[api_name])

    def add_api(self, api_data):
        """Adds an api to the resource map.

    Args:
      api_data: Data for api being added. Must be wrapped in an ApiData object.

    Raises:
      ApiAlreadyExistsError: API already exists in resource map.
      UnwrappedDataException: API data attempting to be added without being
        wrapped in an ApiData wrapper object.
    """
        if not isinstance(api_data, ApiData):
            raise UnwrappedDataException('Api', api_data)
        elif api_data.get_api_name() in self._resource_map_data:
            raise ApiAlreadyExistsError(api_data.get_api_name())
        else:
            self._resource_map_data.update(api_data.to_dict())

    def update_api(self, api_data):
        """Updates an API's data with the provided api data.

    Args:
      api_data: API Data to update the api with. Must be provided as an ApiData
      object.

    Raises:
      ApiNotFoundError: Api to be updated does not exist.
      UnwrappedDataException: API data attempting to be added without being
        wrapped in an ApiData wrapper object.
    """
        if not isinstance(api_data, ApiData):
            raise UnwrappedDataException('Api', api_data)
        if api_data.get_api_name() not in self._resource_map_data:
            raise ApiNotFoundError(api_data.get_api_name())
        else:
            self._resource_map_data.update(api_data.to_dict())

    def remove_api(self, api_name):
        """Removes an API from the resource map."""
        if api_name not in self._resource_map_data:
            raise ApiNotFoundError(api_name)
        del self._resource_map_data[api_name]

    def export(self, file_path=None):
        """Public method to export resource map to file."""
        self._export_resource_map(file_path)