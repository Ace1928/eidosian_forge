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
class MetadataNotFoundError(ResourceMapError):
    """Exception for when a metadata field does not exist in the Resource."""

    def __init__(self, resource_name, metadata_field):
        super(MetadataNotFoundError, self).__init__('[{}] metadata field not found in [{}] Resource.'.format(metadata_field, resource_name))