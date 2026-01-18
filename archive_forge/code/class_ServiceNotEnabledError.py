from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class ServiceNotEnabledError(exceptions.Error):
    """An error raised when a necessary API is not enabled."""

    def __init__(self, service_friendly_name, service_name, project_id):
        message = '{} has not been used in project {project_id} before or it is disabled. Enable it by visiting https://console.developers.google.com/apis/api/{}/overview?project={project_id} then retry. If you enabled this API recently, wait a few minutes for the action to propagate to our systems and retry.'.format(service_friendly_name, service_name, project_id=project_id)
        super(ServiceNotEnabledError, self).__init__(message)