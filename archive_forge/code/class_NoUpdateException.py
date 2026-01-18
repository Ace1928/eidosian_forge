from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class NoUpdateException(exceptions.Error):
    """Error thrown when an update command is run resulting in no updates."""