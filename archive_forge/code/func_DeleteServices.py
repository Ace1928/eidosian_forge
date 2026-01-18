from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.app import operations_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.util import text
import six
def DeleteServices(api_client, services):
    """Delete the given services."""
    errors = {}
    for service in services:
        try:
            operations_util.CallAndCollectOpErrors(api_client.DeleteService, service.id)
        except operations_util.MiscOperationError as err:
            errors[service.id] = six.text_type(err)
    if errors:
        printable_errors = {}
        for service_id, error_msg in errors.items():
            printable_errors[service_id] = '[{0}]: {1}'.format(service_id, error_msg)
        raise ServicesDeleteError('Issue deleting {0}: [{1}]\n\n'.format(text.Pluralize(len(printable_errors), 'service'), ', '.join(list(printable_errors.keys()))) + '\n\n'.join(list(printable_errors.values())))