from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
from googlecloudsdk.core.resource import resource_printer
def ExtractErrorMessage(error_details):
    """Extracts error details from an apitools_exceptions.HttpError.

  Args:
    error_details: a python dictionary returned from decoding an error that
        was serialized to json.

  Returns:
    Multiline string containing a detailed error message suitable to show to a
    user.
  """
    error_message = io.StringIO()
    error_message.write('Error Response: [{code}] {message}'.format(code=error_details.get('code', 'UNKNOWN'), message=error_details.get('message', '')))
    if 'url' in error_details:
        error_message.write('\n' + error_details['url'])
    if 'details' in error_details:
        error_message.write('\n\nDetails: ')
        resource_printer.Print(resources=[error_details['details']], print_format='json', out=error_message)
    return error_message.getvalue()