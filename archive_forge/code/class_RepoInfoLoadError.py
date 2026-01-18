from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class RepoInfoLoadError(DeployError):
    """Indicates a failure to load a source context file."""

    def __init__(self, filename, inner_exception):
        super(RepoInfoLoadError, self).__init__()
        self.filename = filename
        self.inner_exception = inner_exception

    def __str__(self):
        return 'Could not read repo info file {0}: {1}'.format(self.filename, self.inner_exception)