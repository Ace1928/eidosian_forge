from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import textwrap
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
class InvalidCredentialsException(Error):
    """Exceptions to indicate that invalid credentials were found."""