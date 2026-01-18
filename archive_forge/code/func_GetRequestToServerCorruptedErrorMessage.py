from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import exceptions as core_exceptions
def GetRequestToServerCorruptedErrorMessage():
    """Error message for when the request to server failed an integrity check."""
    return 'The request sent to the server was corrupted in-transit. {}'.format(_ERROR_MESSAGE_SUFFIX)