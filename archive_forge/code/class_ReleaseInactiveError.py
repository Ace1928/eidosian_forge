from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class ReleaseInactiveError(exceptions.Error):
    """Error when a release is not deployed to any target."""

    def __init__(self):
        super(ReleaseInactiveError, self).__init__('This release is not deployed to a target in the active delivery pipeline. Include the --to-target parameter to indicate which target to promote to.')