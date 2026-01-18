from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class UnknownSourceError(exceptions.Error):
    """The path exists but points to an unknown file or directory."""

    def __init__(self, path):
        super(UnknownSourceError, self).__init__('[{path}] could not be identified as a valid source directory or file.'.format(path=path))