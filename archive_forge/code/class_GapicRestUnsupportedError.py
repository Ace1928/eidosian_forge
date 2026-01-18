from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.util import api_enablement
from googlecloudsdk.api_lib.util import apis_internal
from googlecloudsdk.api_lib.util import apis_util
from googlecloudsdk.api_lib.util import exceptions as api_exceptions
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.generated_clients.apis import apis_map
import six
class GapicRestUnsupportedError(Error):
    """An error for the unsupported REST transport on GAPIC Clients."""

    def __init__(self):
        super(GapicRestUnsupportedError, self).__init__('REST transport is not yet supported for GAPIC Clients')