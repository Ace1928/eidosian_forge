from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
class PublishPermissionDeniedException(ServiceException):
    """Exception raised when bucket does not have publish permission to a topic.

    This is raised when a custom attempts to set up a notification config to a
    Cloud Pub/Sub topic, but their GCS bucket does not have permission to
    publish to the specified topic.
  """