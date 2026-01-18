from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.ondemandscanning import util as ods_util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import resources
class UnsupportedOS(core_exceptions.Error):
    """Raised when the user attempts to scan from an unsupported operation system.

  Note that this is not the same error as when a user initiates a scan on a
  container image, but that image itself has an unsupported OS. In this case,
  the gcloud command itself is running on an unsupported operation system.
  """