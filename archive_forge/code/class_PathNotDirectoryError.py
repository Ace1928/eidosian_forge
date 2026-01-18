from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class PathNotDirectoryError(ArtifactRegistryError):
    """Raised when a path is not a directory."""