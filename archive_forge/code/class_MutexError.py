from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class MutexError(exceptions.Error):
    """For when two mutually exclusive flags are specified."""