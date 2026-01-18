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
class TrackLevelResourceData(ResourceData):
    """Data wrapper for track-specific resource metadata.

  This data wrapper represents the metadata for a specific release track of a
  resource. Retrieval of metadata will first check for a track level
  specification of the metadata, and if not found will then retrieve the
  top level metadata value.

  Attributes:
    _resource_name: Name of the resource.
    _api_name: Name of the parent api.
    _resource_data: Metadata for the resource.
    _track: Release track for the resource.
    _track_resource_data: Track specific metadata for the resource.
  """

    def __init__(self, resource_name, api_name, resource_data, track):
        self._track = track
        self._track_resource_data = resource_data.get(self._track, {})
        super(TrackLevelResourceData, self).__init__(resource_name, api_name, resource_data)

    def __getattr__(self, metadata_field):
        """Retrieves the track-specific metadata value for the resource.

    If the specified release track does not have a specified value, the parent
    metadata field value for the resource will be returned.

    Args:
      metadata_field: Metadata field value to retrieve

    Returns:
      Metadata field value for the specified release track-specific or the
      parent metadata field.

    Raises:
      MetadataNotFoundError: Metadata field value wasn't found for the specific
      track or for the parent.
      PrivateAttributeNotFoundError: Private attribute doesn't exist in object.
    """
        if metadata_field.startswith('_'):
            raise PrivateAttributeNotFoundError('TrackLevelResourceData', metadata_field)
        else:
            return self.get_metadata(metadata_field)

    def __setattr__(self, metadata_field, value):
        """Sets the specified metadata field to the provided value.

    If the object is not yet instantiated, then standard __setattr__ behavior
    is observed, allowing for proper object intitialization. After
    initialization, the specified metadata field for the release track is set
    to the provided value.

    Args:
      metadata_field: Metadata field to set the value for.
      value: Value to set the specified metadata field to.

    Returns:
      True
    """
        if metadata_field.startswith('_'):
            super(TrackLevelResourceData, self).__setattr__(metadata_field, value)
        elif metadata_field in self._track_resource_data:
            return self.update_metadata(metadata_field, value)
        else:
            return self.add_metadata(metadata_field, value)

    def to_dict(self):
        return {self._resource_name: self._resource_data}

    def get_metadata(self, metadata_field):
        if metadata_field in self._track_resource_data:
            return self._track_resource_data[metadata_field]
        elif metadata_field in self._resource_data:
            return self._resource_data[metadata_field]
        else:
            raise MetadataNotFoundError(self._resource_name, metadata_field)

    def add_metadata(self, metadata_field, value):
        if metadata_field in self._track_resource_data:
            raise MetadataAlreadyExistsError(self._resource_name, metadata_field)
        else:
            self._track_resource_data[metadata_field] = value

    def update_metadata(self, metadata_field, value):
        if metadata_field not in self._track_resource_data:
            raise MetadataNotFoundError(self._resource_name, metadata_field)
        else:
            self._track_resource_data[metadata_field] = value

    def remove_metadata(self, metadata_field):
        if metadata_field not in self._track_resource_data:
            raise MetadataNotFoundError(self._resource_name, metadata_field)
        else:
            del self._track_resource_data[metadata_field]

    def get_release_track(self):
        return self._track

    def get_release_track_data(self, release_track):
        raise TrackLevelResourceReleaseTrackError(release_track, self._track)