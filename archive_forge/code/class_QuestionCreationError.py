from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.command_lib.survey import util as survey_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
import six
class QuestionCreationError(Error):
    """Raises when question cannot be created with the provided data."""

    def __init__(self, required_fields):
        required_fields_in_string = ', '.join(required_fields)
        super(QuestionCreationError, self).__init__('Question cannot be created because either some required field is missing or there are redundant fields. Required fields are {}.'.format(required_fields_in_string))