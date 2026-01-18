from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import os
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage.resources import resource_util
class FileDirectoryResource(Resource):
    """Wrapper for a File system directory."""
    TYPE_STRING = 'file_directory'

    def is_container(self):
        return True