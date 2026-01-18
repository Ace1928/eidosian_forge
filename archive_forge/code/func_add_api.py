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