from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.command_lib.survey import util as survey_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
import six
@classmethod
def FromDictionary(cls, content):
    try:
        return cls(**content)
    except TypeError:
        raise QuestionCreationError(required_fields=['question', 'instruction'])