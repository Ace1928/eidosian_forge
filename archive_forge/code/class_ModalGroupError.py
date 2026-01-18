from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class ModalGroupError(Error):
    """Error when a modal group was not specified correctly."""

    def __init__(self, concept_name, specified, missing):
        super(ModalGroupError, self).__init__('Failed to specify [{}]: {specified}: {missing} must be specified.'.format(concept_name, specified=specified, missing=missing))