from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml
import six
class SourcePathIsNotDirectoryError(exceptions.Error):
    """Raised when the source path is not a directory.

  The deploy command validates that the file path provided by the --source
  command line flag is a directory, and if not, raises this exception.
  """

    def __init__(self, src_path):
        msg = 'Source path is not a directory: {}'.format(src_path)
        super(SourcePathIsNotDirectoryError, self).__init__(msg)