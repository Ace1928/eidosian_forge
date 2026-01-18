from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.logging import util
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import times
import six
def _ExtractServiceAndVersion(entry):
    """Extract service and version from a App Engine log entry.

  Args:
    entry: An App Engine log entry.

  Returns:
    A 2-tuple of the form (service_id, version_id)
  """
    ad_prop = entry.resource.labels.additionalProperties
    service = next((x.value for x in ad_prop if x.key == 'module_id'))
    version = next((x.value for x in ad_prop if x.key == 'version_id'))
    return (service, version)