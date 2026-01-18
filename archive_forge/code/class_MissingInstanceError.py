from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class MissingInstanceError(exceptions.Error):
    """An instance required for the operation does not exist."""

    def __init__(self, instance):
        super(MissingInstanceError, self).__init__('Instance [{}] does not exist.'.format(instance))