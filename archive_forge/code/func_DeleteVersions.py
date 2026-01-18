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
def DeleteVersions(api_client, versions):
    """Delete the given version of the given services."""
    errors = {}
    for version in versions:
        version_path = '{0}/{1}'.format(version.service, version.id)
        try:
            operations_util.CallAndCollectOpErrors(api_client.DeleteVersion, version.service, version.id)
        except operations_util.MiscOperationError as err:
            errors[version_path] = six.text_type(err)
    if errors:
        printable_errors = {}
        for version_path, error_msg in errors.items():
            printable_errors[version_path] = '[{0}]: {1}'.format(version_path, error_msg)
        raise VersionsDeleteError('Issue deleting {0}: [{1}]\n\n'.format(text.Pluralize(len(printable_errors), 'version'), ', '.join(list(printable_errors.keys()))) + '\n\n'.join(list(printable_errors.values())))