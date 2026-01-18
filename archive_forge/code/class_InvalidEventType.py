from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class InvalidEventType(exceptions.Error):
    """Error when a given event type is invalid."""