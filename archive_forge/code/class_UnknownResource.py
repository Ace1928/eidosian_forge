from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import os
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage.resources import resource_util
class UnknownResource(Resource):
    """Represents a resource that may or may not exist."""
    TYPE_STRING = 'unknown'

    def is_container(self):
        raise errors.ValueCannotBeDeterminedError('Unknown whether or not UnknownResource is a container.')