from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class MissingClusterField(exceptions.Error):
    """Class for errors by missing cluster fields."""

    def __init__(self, cluster_id, field, extra_message=None):
        message = 'Cluster {} is missing {}.'.format(cluster_id, field)
        if extra_message:
            message += ' ' + extra_message
        super(MissingClusterField, self).__init__(message)