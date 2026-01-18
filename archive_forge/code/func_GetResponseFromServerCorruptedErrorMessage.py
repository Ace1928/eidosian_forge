from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import exceptions as core_exceptions
def GetResponseFromServerCorruptedErrorMessage():
    """Error message for when the response from server failed an integrity check."""
    return 'The response received from the server was corrupted in-transit. {}'.format(_ERROR_MESSAGE_SUFFIX)