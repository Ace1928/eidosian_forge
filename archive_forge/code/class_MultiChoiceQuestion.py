from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.command_lib.survey import util as survey_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
import six
class MultiChoiceQuestion(Question):
    """Multi-choice question.

  Attributes:
    _choices: [str], list of choices.
  """

    def __init__(self, question, instruction, instruction_on_rejection, choices, answer=None):
        super(MultiChoiceQuestion, self).__init__(question, instruction, instruction_on_rejection, answer=answer)
        self._choices = choices

    @classmethod
    def FromDictionary(cls, content):
        try:
            return cls(**content)
        except TypeError:
            raise QuestionCreationError(required_fields=['question', 'instruction', 'instruction_on_rejection', 'choices'])

    def _PrintQuestion(self):
        """Prints question and lists all choices."""
        question_repr = self._FormatQuestion(indexes=range(1, len(self._choices) + 1))
        log.Print(question_repr)

    def _FormatQuestion(self, indexes):
        """Formats question to present to users."""
        choices_repr = ['[{}] {}'.format(index, msg) for index, msg in zip(indexes, self._choices)]
        choices_repr = [survey_util.Indent(content, 2) for content in choices_repr]
        choices_repr = '\n'.join(choices_repr)
        question_repr = survey_util.Indent(self._question, 1)
        return '\n'.join([question_repr, choices_repr])

    def AcceptAnswer(self, answer):
        """Returns True if answer is accepted, otherwise returns False."""
        try:
            answer_int = int(answer)
        except ValueError:
            return False
        else:
            return 1 <= answer_int <= len(self._choices)

    def Choice(self, index):
        """Gets the choice at the given index."""
        return self._choices[index - 1]

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self._question == other._question and self._instruction == other._instruction and (self._instruction_on_rejection == other._instruction_on_rejection) and (self._choices == other._choices)
        return False

    def __hash__(self):
        return hash((self._question, self._instruction, self._instruction_on_rejection, tuple(self._choices)))

    def __len__(self):
        return len(self._choices)