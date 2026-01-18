from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.app import env
from googlecloudsdk.api_lib.app import metric_names
from googlecloudsdk.api_lib.app import operations_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core.util import retry
from googlecloudsdk.core.util import text
from googlecloudsdk.core.util import times
import six
from six.moves import map  # pylint: disable=redefined-builtin
def GetVersionResource(self, api_client):
    """Attempts to load the Version resource for this version.

    Returns the cached Version resource if it exists. Otherwise, attempts to
    load it from the server. Errors are logged and ignored.

    Args:
      api_client: An AppengineApiClient.

    Returns:
      The Version resource, or None if it could not be loaded.
    """
    if not self.version:
        try:
            self.version = api_client.GetVersionResource(self.service, self.id)
            if not self.version:
                log.info('Failed to retrieve resource for version [{0}]'.format(self))
        except apitools_exceptions.Error as e:
            log.warning('Error retrieving Version resource [{0}]: {1}'.format(six.text_type(self), six.text_type(e)))
    return self.version