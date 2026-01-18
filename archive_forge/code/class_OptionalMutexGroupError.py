from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class OptionalMutexGroupError(Error):
    """Error when an optional mutex group was not specified correctly."""

    def __init__(self, concept_name, conflict):
        super(OptionalMutexGroupError, self).__init__('Failed to specify [{}]: At most one of {conflict} can be specified.'.format(concept_name, conflict=conflict))