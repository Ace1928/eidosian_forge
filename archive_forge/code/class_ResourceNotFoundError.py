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
class ResourceNotFoundError(ResourceMapError):
    """Exception for when a Resource does not exist in the API."""

    def __init__(self, resource_name):
        super(ResourceNotFoundError, self).__init__('[{}] resource not found in ResourceMap.'.format(resource_name))