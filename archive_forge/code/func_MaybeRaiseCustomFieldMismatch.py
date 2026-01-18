from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import re
from googlecloudsdk.api_lib.util import exceptions as exceptions_util
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.core import exceptions
import six
def MaybeRaiseCustomFieldMismatch(error, help_text=''):
    """Special handling for port field type mismatch.

  Due to differences in golang structs used by clusters and proto messages used
  by gcloud, some invalid service responses should be specially handled.
  See b/149365868#comment5 for more info.

  Args:
    error: original error complaining of a type mismatch.
    help_text: str, a descriptive message to help with understanding the error.

  Raises:
    FieldMismatchError: If the error is due to our own custom handling or the
      original error if not.
  """
    regex_match = VALIDATION_ERROR_MSG_REGEX.match(str(error))
    if regex_match:
        if regex_match.group(1) == 'port':
            raise FieldMismatchError('Error decoding the "port" field. Only integer ports are supported by gcloud. Please change your port from "{}" to an integer value to be compatible with gcloud.'.format(regex_match.group(2)))
        elif regex_match.group(1) == 'value':
            raise FieldMismatchError('{0}\n{1}'.format(six.text_type(error), help_text))
    raise error