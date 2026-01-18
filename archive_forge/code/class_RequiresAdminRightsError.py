from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import sys
import traceback
from googlecloudsdk.core.util import encoding
import six
class RequiresAdminRightsError(Error):
    """An exception for when you don't have permission to modify the SDK.

  This tells the user how to run their command with administrator rights so that
  they can perform the operation.
  """

    def __init__(self, sdk_root):
        from googlecloudsdk.core.util import platforms
        message = 'You cannot perform this action because you do not have permission to modify the Google Cloud SDK installation directory [{root}].\n\n'.format(root=sdk_root)
        if platforms.OperatingSystem.Current() == platforms.OperatingSystem.WINDOWS:
            message += 'Click the Google Cloud SDK Shell icon and re-run the command in that window, or re-run the command with elevated privileges by right-clicking cmd.exe and selecting "Run as Administrator".'
        else:
            gcloud_path = os.path.join(sdk_root, 'bin', 'gcloud')
            message += 'Re-run the command with sudo: sudo {0} ...'.format(gcloud_path)
        super(RequiresAdminRightsError, self).__init__(message)