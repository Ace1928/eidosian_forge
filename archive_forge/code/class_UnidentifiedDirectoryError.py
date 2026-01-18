from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from gae_ext_runtime import ext_runtime
from googlecloudsdk.api_lib.app import ext_runtime_adapter
from googlecloudsdk.api_lib.app.runtimes import python
from googlecloudsdk.api_lib.app.runtimes import python_compat
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
class UnidentifiedDirectoryError(exceptions.Error):
    """Raised when GenerateConfigs() can't identify the directory."""

    def __init__(self, path):
        """Constructor.

    Args:
      path: (basestring) Directory we failed to identify.
    """
        super(UnidentifiedDirectoryError, self).__init__('Unrecognized directory type: [{0}]'.format(path))
        self.path = path