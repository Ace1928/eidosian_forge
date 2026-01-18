from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
import time
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core.util import platforms
from googlecloudsdk.core.util import semver
import six
def NotificationMessage(self):
    """Gets the notification message to print to the user.

    Returns:
      str, The notification message the user should see.
    """
    if self.custom_message:
        msg = self.custom_message
    else:
        msg = self.annotation + '\n\n' if self.annotation else ''
        if self.update_to_version:
            version_string = ' --version ' + self.update_to_version
        else:
            version_string = ''
        msg += 'Updates are available for some Google Cloud CLI components.  To install them,\nplease run:\n  $ gcloud components update{version}'.format(version=version_string)
    return '\n\n' + msg + '\n\n'