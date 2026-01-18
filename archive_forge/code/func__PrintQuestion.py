from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.command_lib.survey import util as survey_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
import six
def _PrintQuestion(self):
    question = survey_util.Indent(self._question, 1)
    log.Print(question)