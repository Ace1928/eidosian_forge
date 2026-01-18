from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.ondemandscanning import util as ods_util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import resources
class ExtractionFailedError(core_exceptions.Error):
    """Raised when extraction fails."""
    pass