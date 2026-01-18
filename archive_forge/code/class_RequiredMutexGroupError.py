from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class RequiredMutexGroupError(Error):
    """Error when a required mutex group was not specified correctly."""

    def __init__(self, concept_name, conflict):
        super(RequiredMutexGroupError, self).__init__('Failed to specify [{}]: Exactly one of {conflict} must be specified.'.format(concept_name, conflict=conflict))