from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class ConstraintError(Error):
    """Error when converting a constraint."""

    def __init__(self, concept_name, kind, string, message):
        super(ConstraintError, self).__init__('Invalid {} [{}] for [{}]. {}'.format(kind, string, concept_name, message))