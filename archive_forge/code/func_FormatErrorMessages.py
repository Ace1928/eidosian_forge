from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def FormatErrorMessages(exception):
    """Format app profile error message from API and raise new exception.

  The error messages returned from the backend API are not formatted well when
  using the default format. This raises a new generic exception with a well
  formatted error message built from the original response.

  Args:
    exception: HttpError raised by API.

  Raises:
    exceptions.HttpException: Reformatted error raised by API.
  """
    response = json.loads(exception.content)
    if response.get('error') is None or response.get('error').get('details') is None:
        raise exception
    errors = ['Errors:']
    warnings = ['Warnings (use --force to ignore):']
    for detail in response['error']['details']:
        violations = detail.get('violations', [])
        for violation in violations:
            if violation.get('type').startswith(WARNING_TYPE_PREFIX):
                warnings.append(violation.get('description'))
            else:
                errors.append(violation.get('description'))
    error_msg = ''
    if len(warnings) > 1:
        error_msg += '\n\t'.join(warnings)
    if len(errors) > 1:
        error_msg += '\n\t'.join(errors)
    if not error_msg:
        raise exception
    raise exceptions.HttpException(exception, '{}\n{}'.format(response['error']['message'], error_msg))