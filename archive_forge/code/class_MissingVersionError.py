from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class MissingVersionError(exceptions.Error):
    """A version required for the operation does not exist."""

    def __init__(self, version):
        super(MissingVersionError, self).__init__('Version [{}] does not exist.'.format(version))