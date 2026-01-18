from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage.tasks import task_util
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
def _get_scheme_value(self, url):
    """Extracts the scheme as an integer value from a storage_url."""
    if url:
        return PROVIDER_PREFIX_TO_METRICS_KEY[url.scheme]
    return UNSET