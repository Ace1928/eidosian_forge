from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class GkeClusterGetError(Error):
    """Error while getting a GKE Cluster."""

    def __init__(self, cause):
        super(GkeClusterGetError, self).__init__('Error while getting the GKE Cluster: {0}'.format(cause))