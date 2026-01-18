from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import enum
from googlecloudsdk.command_lib.survey import question
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import pkg_resources
class Survey(object):
    """The survey class.

  Survey content are defined in yaml files in
  googlecloudsdk/command_lib/survey/contents. Each yaml file represents one
  survey.

  Attributes:
    name: str, name of the survey. It should match a name of one yaml file in
      googlecloudsdk/command_lib/survey/contents (w/o the file extension).
    _survey_content: parsed yaml data, raw content of the survey.
    questions: [Question], list of questions in this survey.
    welcome: str, welcome message when entering the survey.
  """

    @enum.unique
    class ControlOperation(enum.Enum):
        EXIT_SURVEY = 'x'
        SKIP_QUESTION = 's'
    INSTRUCTION_MESSAGE = 'To skip this question, type {}; to exit the survey, type {}.'.format(ControlOperation.SKIP_QUESTION.value, ControlOperation.EXIT_SURVEY.value)

    def __init__(self, name):
        self.name = name
        self._survey_content = self._LoadSurveyContent()
        self._questions = list(self._LoadQuestions())

    def _LoadSurveyContent(self):
        """Loads the survey yaml file and return the parsed data."""
        survey_file = os.path.join(_GetSurveyContentDirectory(), self.name + '.yaml')
        survey_data = pkg_resources.GetResourceFromFile(survey_file)
        return yaml.load(survey_data)

    def _LoadQuestions(self):
        """Generator of questions in this survey."""
        for q in self._survey_content['questions']:
            question_type = q['question_type']
            if not hasattr(question, question_type):
                raise QuestionTypeNotDefinedError('The question type is not defined.')
            yield getattr(question, question_type).FromDictionary(q['properties'])

    @property
    def questions(self):
        return self._questions

    @property
    def welcome(self):
        return self._survey_content['welcome']

    def PrintWelcomeMsg(self):
        log.err.Print(self.welcome)

    @classmethod
    def PrintInstruction(cls):
        log.err.Print(cls.INSTRUCTION_MESSAGE)

    def __iter__(self):
        return iter(self._questions)