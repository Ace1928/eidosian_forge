from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.core import exceptions
import six
class CreatingHttpAndAppEngineQueueError(exceptions.InternalError):
    """Error for when attempt to create a queue with both http and App Engine targets."""